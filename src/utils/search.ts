export interface SearchDocument {
    slug: string;
    category: string;
    title: string;
    description: string;
    tags: string[];
    body: string;
}

const normalize = (text: string) => text.normalize("NFKC").toLowerCase();

export const searchPosts = (documents: SearchDocument[], query: string): SearchDocument[] => {
    const terms = normalize(query).trim().split(/\s+/).filter(Boolean);
    if (terms.length === 0) return [];

    return documents
        .map((document) => {
            const fields = [
                { text: normalize(document.title), weight: 4 },
                { text: normalize(document.tags.join(" ")), weight: 3 },
                { text: normalize(document.description), weight: 2 },
                { text: normalize(document.body), weight: 1 },
            ];
            let score = 0;
            for (const term of terms) {
                const matches = fields.filter(({ text }) => text.includes(term));
                if (matches.length === 0) return { document, score: 0 };
                score += matches.reduce((sum, { weight }) => sum + weight, 0);
            }
            return { document, score };
        })
        .filter(({ score }) => score > 0)
        .sort((a, b) => b.score - a.score)
        .map(({ document }) => document);
};
