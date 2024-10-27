// import { defineConfig } from 'vite';
// import wasm from "vite-plugin-wasm";
// // import topLevelAwait from "vite-plugin-top-level-await";
// //
// export default defineConfig({
//     build: {
//         target: 'esnext',  // Ensures that newer syntax is preserved.
//         rollupOptions: {
//             // Ensures that WASM files are handled properly.
//             output: {
//                 entryFileNames: '[name].[hash].js',
//                 chunkFileNames: '[name].[hash].js',
//                 assetFileNames: '[name].[hash][extname]',
//             }
//         },
//     },
//     plugins: [
//         wasm(),
//         // topLevelAwait()
//     ],
//     worker: {
//         // Not needed with vite-plugin-top-level-await >= 1.3.0
//         // format: "es",
//         plugins: [
//             wasm(),
//             // topLevelAwait()
//         ]
//     }
// })