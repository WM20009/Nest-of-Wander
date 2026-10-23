import Redis from 'ioredis'
import { getDateKeyInShanghai, type ViewStats } from "./shared";

const client = new Redis(import.meta.env.REDIS_URI)

export const getViewsBySlug = async (slug: string): Promise<ViewStats> => {
	if (!slug) {
		throw new Error("A valid slug is required to collect post stats.");
	}

	const dateKey = getDateKeyInShanghai(new Date());
	const totalKey = `post_views_total:${slug}`;
	const dailyKey = `post_views_daily:${slug}:${dateKey}`;

	const totalViews = await client.incr(totalKey);
	const todayViews = await client.incr(dailyKey);

	return {
		slug,
		todayViews,
		totalViews,
	};
};
