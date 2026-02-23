import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def scrape_patpat_exact(max_pages=200):
    all_vehicles = []

    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )

    for page in range(1, max_pages + 1):
        print(f"\n--- Scraping Patpat.lk page {page} ---")
        url = f"https://patpat.lk/en/sri-lanka/vehicle/car?page={page}"

        try:
            response = scraper.get(url)

            if response.status_code != 200:
                print(f"Blocked! Status code: {response.status_code}")
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            listings = soup.find_all('a', class_=lambda c: c and 'block w-full' in c)

            for item in listings:
               
                price_tag = item.find('span', class_=lambda c: c and 'bg-gradient-to-r' in c)
                price = price_tag.text.strip() if price_tag else "N/A"

                detail_rows = item.find_all('div', class_=lambda c: c and 'flex-row' in c and 'gap-2' in c)
                location = "N/A"
                if len(detail_rows) > 1:
                    spans = detail_rows[1].find_all('span')
                    if len(spans) > 0:
                        location = spans[-1].text.strip()

                ad_url = item.get('href')

                if ad_url and price != "N/A":
                    if not ad_url.startswith('http'):
                        ad_url = "https://patpat.lk" + ad_url
                        
                   
                    mileage, engine, manufacturer, year, model, fuel_type = ["N/A"] * 6
                    
                    try:
                        deep_response = scraper.get(ad_url)
                        deep_soup = BeautifulSoup(deep_response.text, 'html.parser')
                        
                       
                        li_elements = deep_soup.find_all('li', class_=lambda c: c and 'justify-between' in c)
                        
                        for li in li_elements:
                           
                            div_tag = li.find('div')
                            if not div_tag:
                                continue
                            
                            label = div_tag.text.strip().lower()
                            
                            spans = li.find_all('span')
                            if not spans:
                                continue
                                
                            value = spans[-1].text.strip()
                            
                            
                            if label == "mileage":
                                mileage = value
                            elif label == "manufacturer":
                                manufacturer = value
                            elif label == "model year":
                                year = value
                            elif label == "model": 
                                model = value
                            elif label == "fuel type":
                                fuel_type = value
                            elif "engine" in label: 
                                engine = value
                                
                    except Exception as e:
                        pass
                    
                    time.sleep(random.uniform(1.0, 2.5))
                    
                    print(f"Scraped | Price: {price} | Make: {manufacturer} | Model: {model} | Mileage: {mileage}")

                    # 3. COMBINE AND SAVE
                    all_vehicles.append({
                        'Price': price,
                        'Location': location,
                        'Mileage': mileage,
                        'Engine': engine,
                        'Manufacturer': manufacturer,
                        'Year': year,
                        'Model': model,
                        'Fuel_Type': fuel_type
                    })

        except Exception as e:
            print(f"Error on page {page}: {e}")
            break

        time.sleep(random.uniform(2, 5))

    return pd.DataFrame(all_vehicles)

if __name__ == "__main__":
      
    dataset = scrape_patpat_exact(max_pages=200)
    print(dataset.head())

    if not dataset.empty:
        dataset.to_csv('patpat_targeted_data.csv', index=False)
        print(f"\nSuccess! Saved {len(dataset)} rows to CSV.")
