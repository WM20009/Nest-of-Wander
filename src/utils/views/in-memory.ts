import { getDateKeyInShanghai, type ViewStats } from "./shared";

const totalViews = new Map<string, number>();
const dailyViews = new Map<string, number>();

export const getViewsBySlug = async (slug: string): Promise<ViewStats> => {
	if (!slug) {
		throw new Error("A valid slug is required to collect post stats.");
	}

	const dateKey = getDateKeyInShanghai(new Date());
	const dailyKey = `${slug}:${dateKey}`;
	const nextTotalViews = (totalViews.get(slug) ?? 0) + 1;
	const nextDailyViews = (dailyViews.get(dailyKey) ?? 0) + 1;

	totalViews.set(slug, nextTotalViews);
	dailyViews.set(dailyKey, nextDailyViews);

	return {
		slug,
		todayViews: nextDailyViews,
		totalViews: nextTotalViews,
	};
};
