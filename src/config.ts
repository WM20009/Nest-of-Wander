import type { NavItems } from "./types";

export const NAV_ITEMS: NavItems = {
	home: {
		path: "/",
		title: "home",
	},
	blog: {
		path: "/blog",
		title: "blog",
	},
	tags: {
		path: "/tags",
		title: "tags",
	},
	media: {
		path: "/media",
		title: "media",
	},
	about: {
		path: "/about",
		title: "about",
	},
};

export const SITE = {
	// Your site's detail?
	name: "Wander 的小窝",
	title: "I'm wandering",
	description: "这里是我的点点滴滴",
	url: "https://astro-ink.vercel.app",
	githubUrl: "https://github.com/WM20009",
	listDrafts: true,
	image:
		"https://raw.githubusercontent.com/one-aalam/astro-ink/main/public/astro-banner.png",
	// YT video channel Id (used in media.astro)
	ytChannelId: "",
	// Optional, user/author settings (example)
	// Author: name
	author: "Wander", // Example: Fred K. Schott
	// Author: Twitter handler
	authorGithub: "WM20009", // Example: FredKSchott
	// Author: Image external source
	authorImage: "/images/avatar.jpg", // Example: https://pbs.twimg.com/profile_images/1272979356529221632/sxvncugt_400x400.jpg, https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png
	// Author: Bio
	authorBio:
		"Xjtu 24级计试，目前在研究具身智能相关话题~",
};

// Ink - Theme configuration
export const PAGE_SIZE = 8;
export const USE_POST_IMG_OVERLAY = false;
export const USE_MEDIA_THUMBNAIL = true;

export const USE_AUTHOR_CARD = true;
export const USE_SUBSCRIPTION = false; /* works only when USE_AUTHOR_CARD is true */

export const USE_VIEW_STATS = false;
