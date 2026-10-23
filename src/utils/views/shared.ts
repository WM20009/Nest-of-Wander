export interface ViewStats {
    slug: string;
    todayViews: number;
    totalViews: number;
}

export const getDateKeyInShanghai = (date: Date) => {
    const formatter = new Intl.DateTimeFormat("en-CA", {
        timeZone: "Asia/Shanghai",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
    });

    return formatter.format(date);
};
