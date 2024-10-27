import init, {RPCLinearRegistryServer} from './pkg';

let SERVER = undefined;

onmessage = async (e) => {
    if (SERVER === undefined) {
        await init();
        SERVER = RPCLinearRegistryServer.wasm_new();
    }
    let output = await SERVER.serve_serialized(e.data);

    postMessage(output);
}
