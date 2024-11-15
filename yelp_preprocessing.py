import json
import pandas as pd

business_json_path = '../../yelp_dataset/yelp_academic_dataset_business.json'
review_json_path = '../../yelp_dataset/yelp_academic_dataset_review.json'
photos_json_path = '../../yelp_photos/photos.json'


output_csv_path = 'businesses.csv'


business_reviews = {}
with open(review_json_path, 'r', encoding='utf-8') as f:
    for line in f:
        review = json.loads(line)
        business_id = review['business_id']
        if business_id not in business_reviews:
            business_reviews[business_id] = review['text']

business_photos = {}
with open(photos_json_path, 'r', encoding='utf-8') as f:
    for line in f:
        photo = json.loads(line)
        business_id = photo['business_id']
     
        if business_id not in business_photos and photo.get('caption'):
            business_photos[business_id] = photo  


processed_data = []
with open(business_json_path, 'r', encoding='utf-8') as f:
    count = 0
    for line in f:
        if count >= 10000:
            break
        business = json.loads(line)
        business_id = business['business_id']
        if business_id in business_reviews and business_id in business_photos:
            photo = business_photos[business_id]
            record = {
                'business_id': business_id,
                'name': business.get('name', ''),
                'address': business.get('address', ''),
                'city': business.get('city', ''),
                'state': business.get('state', ''),
                'stars': business.get('stars', ''),
                'hours': business.get('hours', {}),
                'photo_id': photo.get('photo_id', ''),
                'photo_caption': photo.get('caption', ''),
                'photo_label': photo.get('label', ''),
                'review': business_reviews[business_id]
            }
            processed_data.append(record)
            count += 1


business_df = pd.DataFrame(processed_data)
business_df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Processed {len(processed_data)} business records and saved to {output_csv_path}")
