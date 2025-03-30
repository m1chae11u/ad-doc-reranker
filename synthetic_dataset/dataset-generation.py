import random
from faker import Faker
import pandas as pd

fake = Faker()

subdomains_with_categories = {
    "Electronics": ["smartphones", "laptops", "smartwatches", "headphones", "home-automation"],
    "Fashion": ["menswear", "womenswear", "accessories", "footwear", "jewelry"],
    "Food": ["snacks", "organic-food", "beverages", "frozen-food", "desserts"],
    "Health & Wellness": ["fitness", "supplements", "mental-health", "skincare", "nutrition"],
    "Services": ["home-services", "cleaning-services", "IT-support", "education-services", "financial-advisory"],
    "Travel": ["flights", "hotels", "vacation-packages", "car-rentals", "cruises"],
    "Home & Garden": ["furniture", "gardening-tools", "home-decor", "outdoor-furniture", "kitchen-appliances"],
    "Automotive": ["cars", "motorcycles", "car-accessories", "electric-vehicles", "car-insurance"],
    "Entertainment": ["movies", "games", "books", "music", "streaming-services"],
    "Makeup": ["mascara", "foundation", "lipstick", "eyeliner", "blush"]
}

# 50 products (1 / subdomain)
products = [
    "SuperPhone X", "Ultra Laptop", "Smartwatch 360", "Noise Cancelling Headphones", "Smart Home Hub",
    "Menswear Collection", "Womenswear Fashion", "Premium Accessories", "Trendy Footwear", "Luxury Jewelry",
    "Gourmet Snacks", "Organic Coffee Beans", "Healthy Beverages", "Frozen Vegetables", "Gourmet Desserts",
    "Fitness Tracker", "Vitamins and Supplements", "Mental Wellness App", "Organic Skincare Set", "Healthy Meal Plans",
    "Home Services Package", "Cleaning Service Pro", "IT Support Plan", "Online Learning Services", "Financial Consulting",
    "Airline Tickets", "Luxury Hotels", "Exclusive Vacation Packages", "Car Rentals Deals", "Cruise Packages",
    "Designer Furniture", "Garden Tools", "Modern Home Decor", "Outdoor Furniture Set", "Smart Kitchen Appliances",
    "Electric Cars", "Motorcycles for Rent", "Car Accessories Shop", "Electric Vehicle Deals", "Car Insurance Plans",
    "Latest Movies", "Video Games", "Best-Selling Books", "Music Album Releases", "Streaming TV Shows",
    "Mascara Pro", "Flawless Foundation", "Long-lasting Lipstick", "Waterproof Eyeliner", "Natural Blush"
]

# metadata
brands = ["Brand A", "Brand B", "Brand C"]
locations = ["USA", "UK", "Germany", "Canada"]

ad_templates = [
    "Get your hands on {product} today! {category} at unbeatable prices. Don't miss out!",
    "Looking for the best {category}? Buy {product} now and enjoy amazing features!",
    "Limited-time offer! Grab {product} in the {category} category for an exclusive deal!",
    "Transform your {category} experience with {product}. Get yours today and save big!",
    "Shop {product} and make your {category} experience unforgettable. Order now!",
    "Get {product} now! The perfect {category} solution you’ve been waiting for.",
    "Don’t miss out on {product} - the ultimate {category} product. Act fast!",
    "Revolutionize your {category} with {product}. Get yours today for the best price!",
    "Discover {product}, your new go-to for {category}. Buy now and get a great deal!",
    "Shop {product} in the {category} category. Perfect for upgrading your experience!"
]

def generate_unique_ad(product, category):
    template = random.choice(ad_templates)
    ad = template.format(product=product, category=category)
    return ad

dataset = []

product_index = 0
for large_domain, subdomains in subdomains_with_categories.items():
    for subdomain in subdomains:
        product = products[product_index]
        for _ in range(100):  # 100 ads / subdomain
            category = large_domain
            ad_copy = generate_unique_ad(product, subdomain)
            url = f"https://example.com/{subdomain.replace(' ', '-').lower()}"
            price = f"${random.randint(10, 500)}"  #random price - fix later for accuracy
            user_reviews = f"{random.randint(100, 5000)} reviews - {random.randint(1, 5)} stars"  # Random reviews and ratings
            brand = random.choice(brands)
            location = random.choice(locations)
            image_url = f"https://example.com/images/{product.replace(' ', '-').lower()}.jpg"

            dataset.append([product, ad_copy, url, price, category, user_reviews, f"Brand: {brand}, Location: {location}, Image: {image_url}"])

        product_index += 1  

df = pd.DataFrame(dataset, columns=["Product/Service Name", "Description/Ad Copy", "URL", "Price", "Category/Domain", "User Reviews/Ratings", "Metadata"])

df.to_csv('ad_documents.csv', index=False)

print("Dataset generated and saved to 'ad_documents.csv'")
print(df.head())