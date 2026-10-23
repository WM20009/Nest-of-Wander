import type { APIRoute } from "astro";
import { getEntryBySlug } from "astro:content";
import { getViewsBySlug } from "../../../../utils/views/turso";

export const prerender = false;

const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), {
        status,
        headers: {
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "no-store",
        },
    });

export const GET: APIRoute = async ({ params }) => {
    const slug = (params.slug ?? "").trim();

    if (!slug) {
        return json({ error: "Missing blog slug." }, 400);
    }

    const post = await getEntryBySlug("blog", slug);

    if (!post) {
        return json({ error: "Post not found.", slug }, 404);
    }

    try {
        const stats = await getViewsBySlug(slug);
        return json(stats);
    } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to collect post stats.";
        return json({ error: message, slug }, 500);
    }
};
