from google_play_scraper import Sort, reviews
import csv
from datetime import datetime
import time
import logging

logging.basicConfig(filename='scraper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_google_play_reviews(app_id, bank_name, target_count=400):
    logging.info(f"Fetching reviews for {bank_name} (App ID: {app_id})...")
    all_reviews_for_app = []
    continuation_token = None
    max_retries = 5
    current_retry = 0

    while len(all_reviews_for_app) < target_count and current_retry < max_retries:
        try:
            results_batch, new_continuation_token = reviews(
                app_id,
                lang='en',
                country='et',
                sort=Sort.NEWEST,
                count=min(200, target_count - len(all_reviews_for_app)),
                continuation_token=continuation_token
            )
            if not results_batch:
                break
            all_reviews_for_app.extend(results_batch)
            continuation_token = new_continuation_token
            if len(results_batch) < min(200, target_count - len(all_reviews_for_app)) and not continuation_token:
                break
            current_retry = 0
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error during batch scraping for {bank_name} ({app_id}): {e}")
            current_retry += 1
            time.sleep(current_retry * 5)
            if current_retry >= max_retries:
                break

    prepared_reviews = []
    for entry in all_reviews_for_app:
        prepared_reviews.append({
            'review_text': entry.get('content', ''),
            'rating': entry.get('score'),
            'date': entry.get('at').strftime('%Y-%m-%d') if entry.get('at') else '',
            'bank_name': bank_name,
            'source': 'Google Play'
        })
    return prepared_reviews

if __name__ == "__main__":
    app_details = {
        'Commercial Bank of Ethiopia': 'com.cbe.mobilebanking',
        'Bank of Abyssinia': 'com.boa.boaMobileBanking',
        'Dashen Bank': 'com.dashen.dashensuperapp'
    }
    all_collected_reviews = []
    target_reviews_per_bank = 400

    for bank_name, app_id in app_details.items():
        all_collected_reviews.extend(scrape_google_play_reviews(app_id, bank_name, target_reviews_per_bank))
        time.sleep(5)

    if all_collected_reviews:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'raw_reviews_{timestamp}.csv'
        fieldnames = ['review_text', 'rating', 'date', 'bank_name', 'source']
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_collected_reviews)
        print(f"Total raw reviews scraped: {len(all_collected_reviews)}")
        print(f"Raw data saved to {filename}")
