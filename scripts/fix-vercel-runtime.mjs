import { readFile, writeFile } from 'node:fs/promises';

// Astro 4's Vercel adapter emits Node 18 metadata for Node 22 builds.
// Keep its generated handler and dependencies, and use the supported runtime.
const configPath = new URL('../.vercel/output/functions/render.func/.vc-config.json', import.meta.url);
const config = JSON.parse(await readFile(configPath, 'utf8'));

if (!config.runtime?.startsWith('nodejs')) {
    throw new Error('Expected a Vercel Node.js function for the article counter.');
}

config.runtime = 'nodejs22.x';
await writeFile(configPath, JSON.stringify(config, null, 2) + '\n');
console.log('Vercel article counter configured for Node.js 22.');
