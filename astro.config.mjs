import path, { dirname } from 'path';
import { fileURLToPath } from 'url';
import svelte from '@astrojs/svelte';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import mdx from '@astrojs/mdx';
import { defineConfig } from "astro/config";
import vercel from "@astrojs/vercel/serverless";
import markdoc from "@astrojs/markdoc";
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
import remarkCodeTitles from 'remark-code-titles'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
//import decapCmsOauth from "astro-decap-cms-oauth";

// Full Astro Configuration API Documentation:
// https://docs.astro.build/reference/configuration-reference

// https://astro.build/config
export default defineConfig( /** @type {import('astro').AstroUserConfig} */{
  output: 'static',
  site: 'https://astro-ink.vercel.app', // Your public domain, e.g.: https://my-site.dev/. Used to generate sitemaps and canonical URLs.
  server: {
    // port: 4321, // The port to run the dev server on.
  },
  image: {
    service: {
      entrypoint: 'astro/assets/services/noop' // 禁用图片优化
    }
  },
  markdown: {
    syntaxHighlight: 'shiki',
    shikiConfig: {
      theme: 'css-variables',
    },
    remarkPlugins: [
      remarkCodeTitles
    ],
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex]
  },
  integrations: [
    mdx(), 
    markdoc(),
    svelte(), 
    tailwind({
      applyBaseStyles: false,
    }), 
    sitemap(),
    //decapCmsOauth()
  ],
  vite: {
    plugins: [],
    resolve: {
      alias: {
        $: path.resolve(__dirname, './src')
      }
    },
    optimizeDeps: {
      allowNodeBuiltins: true
    }
  },
  //adapter: vercel()
});