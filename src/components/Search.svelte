<script lang="ts">
    import { onMount } from 'svelte'
    import SearchIcon from './SearchIcon.svelte'
    import PostSearchPreview from './PostSearchPreview.svelte'
    import { searchPosts, type SearchDocument } from '../utils/search'

    let searchInput: HTMLInputElement
    let searchableDocs: SearchDocument[] = []
    let loading = true
    let loadError = false

    let searchQuery = ''
    let searchResults: SearchDocument[] = []

    onMount(async() => {
        searchInput.focus()
        try {
            const resp = await fetch('/search-index.json')
            if (!resp.ok) throw new Error(`Search data request failed: ${resp.status}`)
            searchableDocs = await resp.json()
        } catch (error) {
            loadError = true
            console.error('Search', error)
        } finally {
            loading = false
        }
    })

    $: searchResults = searchPosts(searchableDocs, searchQuery)
</script>
<div class="search">
    <div class="search__ctrl">
        <label for="search"><SearchIcon found={searchResults.length > 0} /><span class="sr-only">搜索文章</span></label>
        <input id="search" type="search" name="search" bind:this={searchInput} placeholder="搜索标题、标签或正文" bind:value={searchQuery} />
    </div>
    <div class="search__results">
        {#if searchResults.length}
            {#each searchResults as post, i }
                <PostSearchPreview post={post} isLast={ i === searchResults.length - 1 } />
            {/each}
        {:else}
            <div class="search__results--none">
                {#if loading}
                    正在加载文章…
                {:else if loadError}
                    搜索暂时无法加载，请关闭后重试。
                {:else if searchQuery.trim().length}
                    没有找到相关文章。
                {:else}
                    输入关键词搜索文章。
                {/if}
            </div>
        {/if}
    </div>
    <div class="note"><small>点击外部区域或按 Esc 关闭</small></div>
</div>
<style>
    .search {
        @apply w-full relative bg-theme-primary  p-8  rounded-md shadow-lg;
    }
    input {
        @apply w-full px-4 py-2 pl-10 text-xl font-semibold text-gray-600 border-0 shadow-inner rounded-md bg-gray-100 placeholder-theme-dark-secondary;
    }
    .search__ctrl {
        @apply pb-4 relative;
    }
    .search__ctrl label {
        @apply text-theme-primary absolute top-2 left-2;
    }
    .search__results {
        @apply w-full h-64 py-4 overflow-y-auto;
    }
    .search__results--none {
        @apply  text-center text-theme-dark-primary;
    }
    .note {
        @apply w-full text-center text-white;
    }
</style>
