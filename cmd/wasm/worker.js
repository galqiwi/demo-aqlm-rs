import init, {LlamaLoader} from './pkg/wasm.js';

let LLAMA = undefined;

onmessage = async (e) => {
    let oldMessages = e.data;

    if (LLAMA === undefined) {
        await init();

        let workers = [];
        for (let i = 0; i < navigator.hardwareConcurrency; i++) {
            let worker = new Worker(new URL('./linear_worker.js', import.meta.url));
            workers.push(worker)
        }

        let loader = await LlamaLoader.new(workers);

        let worker_response_senders = loader.get_worker_response_senders();

        for (let i = 0; i < workers.length; i++) {
            let worker = workers[i];
            let response_sender = worker_response_senders[i];
            worker.onmessage = async (e) => {
                await response_sender.register_response(e.data);
            };
        }

        let status_sender = loader.get_download_status_sender();
        let status_callback = (async () => {
            while (true) {
                let newMessages = oldMessages.slice();
                let status = await status_sender.get_status();
                if (status === "cyanide") {
                    break
                }
                newMessages.push({
                    'role': 'Assistant',
                    'content': status,
                })
                postMessage({
                    'messages': newMessages,
                    'is_finished': false,
                });
            }
        })();

        LLAMA = await loader.into_llama_api();

        await status_callback
    }

    if (oldMessages.length === 0) {
        LLAMA.clear();
        postMessage({
            'messages': [],
            'is_finished': true,
        });
        return
    }

    let newMessages = oldMessages.slice();
    newMessages.push({
        'role': 'Assistant',
        'content': 'Processing...',
    })
    postMessage({
        'messages': newMessages,
        'is_finished': false,
    });

    let allMessages = [{
        'role': 'System',
        'content': 'You are a helpful chat assistant.',
    }].concat(e.data);

    await LLAMA.set_prefix(allMessages.map(msg => {
        return JSON.stringify(msg);
    }));

    while (true) {
        let newMessages = (await LLAMA.next()).map(msg => {
            return JSON.parse(msg);
        }).filter(msg => {
            return msg.role !== "System";
        });
        let is_finished = LLAMA.is_finished();

        postMessage({
            'messages': newMessages,
            'is_finished': is_finished,
        });

        if (is_finished) {
            break
        }
    }
};
