import { createClient } from "@libsql/client";
import { getDateKeyInShanghai, type ViewStats } from "./shared";

const tursoUrl = import.meta.env.TURSO_DB_URL;
const tursoAuthToken = import.meta.env.TURSO_DB_AUTH_TOKEN;

let schemaReadyPromise: Promise<void> | null = null;
let client: ReturnType<typeof createClient> | null = null;

const getClient = () => {
    if (!tursoUrl || !tursoAuthToken) {
        throw new Error("Turso is not configured. Please set TURSO_DB_URL and TURSO_DB_AUTH_TOKEN.");
    }

    client ??= createClient({
        url: tursoUrl,
        authToken: tursoAuthToken,
    });
    return client;
};

const ensureSchema = async () => {
    if (!schemaReadyPromise) {
        schemaReadyPromise = (async () => {
            const client = getClient();

            await client.execute(`
                CREATE TABLE IF NOT EXISTS post_views_total (
                    slug TEXT PRIMARY KEY,
                    total_views INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            `);

            await client.execute(`
                CREATE TABLE IF NOT EXISTS post_views_daily (
                    slug TEXT NOT NULL,
                    view_date TEXT NOT NULL,
                    views INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (slug, view_date)
                )
            `);
        })().catch((error) => {
            schemaReadyPromise = null;
            throw error;
        });
    }

    await schemaReadyPromise;
};

export const getViewsBySlug = async (slug: string): Promise<ViewStats> => {
    if (!slug || slug.trim() === "") {
        throw new Error("A valid slug is required to collect post stats.");
    }

    await ensureSchema();

    const client = getClient();
    const now = new Date();
    const timestamp = now.toISOString();
    const viewDate = getDateKeyInShanghai(now);
    const transaction = await client.transaction("write");

    try {
        const totalResult = await transaction.execute({
            sql: `
                INSERT INTO post_views_total (slug, total_views, created_at, updated_at)
                VALUES (:slug, 1, :createdAt, :updatedAt)
                ON CONFLICT(slug) DO UPDATE SET
                    total_views = post_views_total.total_views + 1,
                    updated_at = excluded.updated_at
                RETURNING total_views
            `,
            args: {
                slug,
                createdAt: timestamp,
                updatedAt: timestamp,
            },
        });

        const dailyResult = await transaction.execute({
            sql: `
                INSERT INTO post_views_daily (slug, view_date, views, created_at, updated_at)
                VALUES (:slug, :viewDate, 1, :createdAt, :updatedAt)
                ON CONFLICT(slug, view_date) DO UPDATE SET
                    views = post_views_daily.views + 1,
                    updated_at = excluded.updated_at
                RETURNING views
            `,
            args: {
                slug,
                viewDate,
                createdAt: timestamp,
                updatedAt: timestamp,
            },
        });

        await transaction.commit();

        return {
            slug,
            todayViews: Number(dailyResult.rows[0]?.views ?? 0),
            totalViews: Number(totalResult.rows[0]?.total_views ?? 0),
        };
    } catch (error) {
        try {
            await transaction.rollback();
        } catch (rollbackError) {
            console.error("Failed to roll back Turso view stats", rollbackError);
        }
        console.error("Failed to update Turso view stats", error);
        throw error;
    } finally {
        transaction.close();
    }
};
