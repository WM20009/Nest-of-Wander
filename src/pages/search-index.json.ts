import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import type { SearchDocument } from "../utils/search";

export const prerender = true;

export const GET: APIRoute = async () => {
    const posts = await getCollection("blog");
    const documents: SearchDocument[] = posts.map(({ slug, data, body }) => ({
        slug,
        category: "blog",
        title: data.title,
        description: data.description,
        tags: data.tags,
        body,
    }));

    return new Response(JSON.stringify(documents), {
        headers: { "Content-Type": "application/json; charset=utf-8" },
    });
};
