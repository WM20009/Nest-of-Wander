<script lang="ts">
    import { onMount } from "svelte";

    export let slug: string = ''

    let loading: boolean = false
    let todayViews: number | null = null
    let totalViews: number | null = null

    onMount(async () => {
        if(slug && slug.trim() !== '') {
            try {
                loading = true
                const resp = await fetch(`/api/blog/views/${encodeURIComponent(slug)}.json`)

                if (!resp.ok) {
                    throw new Error(`Failed to fetch post stats: ${resp.status}`)
                }

                const stats = await resp.json()
                todayViews = typeof stats.todayViews === 'number' ? stats.todayViews : null
                totalViews = typeof stats.totalViews === 'number' ? stats.totalViews : null
            } catch(e) {
                console.error('PostStats', e)
            } finally {
                loading = false
            }
        }
    })
</script>
<span class="post-stats__views">
    {#if loading && todayViews === null && totalViews === null}
        今日访问 -- · 总访问 --
    {:else}
        今日访问 {todayViews ?? '--'} · 总访问 {totalViews ?? '--'}
    {/if}
</span>
<style>
    .post-stats__views {
        @apply px-1 mx-1;
    }
</style>
