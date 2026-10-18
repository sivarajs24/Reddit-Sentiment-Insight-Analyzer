import os
from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()

def scrape_reddit(product_name: str, limit_per_sub: int = 50, custom_subreddits: list = None):
    """
    Scrapes Reddit data using Apify's Reddit Scraper.
    Gets top posts and their deep comments.
    """
    token = os.environ.get('APIFY_API_TOKEN')
    if not token:
        print("Error: APIFY_API_TOKEN is missing in .env")
        return []

    client = ApifyClient(token)
    
    # We will just do a general search if custom_subreddits isn't provided.
    search_queries = [product_name]
    if custom_subreddits:
        # Apify actor search supports standard reddit search syntax
        search_queries = [f"{product_name} subreddit:{sub}" for sub in custom_subreddits]
        
    print(f"Starting Apify scraper for queries: {search_queries}")
    
    run_input = {
        "searchQueries": search_queries,
        "type": "post",
        "sort": "relevance",
        "time": "month",
        "maxPostsPerSource": limit_per_sub,
        "includeComments": True,
        "maxCommentsPerPost": 5
    }

    try:
        # Using a reliable, free-to-call actor for Reddit
        run = client.actor("automation-lab/reddit-scraper").call(run_input=run_input)
        
        collected_data = []
        dataset_items = client.dataset(run["defaultDatasetId"]).iterate_items()
        
        for index, item in enumerate(dataset_items):
            # Combine the post body with its top comments
            body_text = item.get('text', '') or item.get('body', '') or ''
            
            comments_text = ""
            for comment in item.get('comments', []):
                text = comment.get('text', '') or comment.get('body', '')
                if text:
                    comments_text += f"\n[Comment]: {text}"
            
            full_content = body_text + "\n" + comments_text
            
            mapped_item = {
                "id": item.get('id', f"apify_{index}"),
                "type": "post",
                "title": item.get('title', ''),
                "body": full_content.strip(),
                "upvotes": item.get('upvotes', 0),
                "subreddit": item.get('subreddit', 'all'),
                "timestamp": 0,
                "datetime": item.get('createdAt', ''),
                "url": item.get('url', '')
            }
            
            if mapped_item["title"] or mapped_item["body"]:
                collected_data.append(mapped_item)
                
        print(f"Apify Scraper finished. Retrieved {len(collected_data)} items (with comments included).")
        return collected_data

    except Exception as e:
        print(f"Apify Scraping Error: {e}")
        return []

if __name__ == "__main__":
    data = scrape_reddit("OpenAI", limit_per_sub=5)
    print(f"Scraped {len(data)} items")
    for d in data:
        print(f"Title: {d['title']}")
        print(f"Body length: {len(d['body'])} chars\n")
