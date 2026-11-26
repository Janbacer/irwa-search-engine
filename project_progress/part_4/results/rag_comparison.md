# RAG comparison (groq / openai/gpt-oss-20b, 2026-11-26 10:38)

|                                                                                 | baseline              | improved   |
|:--------------------------------------------------------------------------------|:----------------------|:-----------|
| relevant recommendation (Q1-Q5, T1)                                             | 4/6                   | 6/6        |
| 'no good products' although relevant products were retrieved (labelled queries) | 3/8                   | 0/8        |
| correct 'no good products' (products the shop does not sell)                    | 5/5                   | 5/5        |
| answers with invented PIDs                                                      | 0/16                  | 0/16       |
| prices, ratings and discounts in the explanations that match the data           | no data in the prompt | 68/68      |
| LLM calls                                                                       | 16                    | 15         |
| median latency (ms)                                                             | 2334                  | 2552       |
| median prompt (characters)                                                      | 2392                  | 4517       |

## Q1: men white formal shirt slim fit

**baseline** (ok, best = SHTFU73NHNAFBCCZ, relevant = 1.0, invented PIDs = 0)

```
- Best Product: **SHTFU73NHNAFBCCZ** Men Slim Fit Solid Spread Collar Formal Shirt  
- Why: This shirt is a true white, slim‑fit formal style with a solid spread collar—ideal for business attire. It balances classic cut, breathable fabric, and a clean look, making it the most straightforward match for a “men white formal shirt slim fit.”  
- Alternative: **SHTFU73NTX6N3R63** Men Slim Fit Solid Spread Collar Formal Shirt (identical style, just a different SKU).
```

**improved** (ok, best = SHTF98P8MTY7ZXAZ, relevant = 1.0, invented PIDs = 0)

```
{"status": "ok", "best": {"pid": "SHTF98P8MTY7ZXAZ", "why": "Men Slim Fit Solid Spread Collar Formal Shirt, white, in stock, highest rating 4.4/5 and reasonable price Rs 799 after 38% discount", "title": "Men Slim Fit Solid Spread Collar Formal Shirt", "rank": 1, "price": 799.0, "rating": 4.4, "in_stock": true}, "alternatives": [{"pid": "SHTFW52ZURHFKNTZ", "why": "White Slim Fit Printed Button Down Collar Formal Shirt, in stock, slightly cheaper Rs 758 and good rating 4.2/5", "title": "Men Slim Fit Printed Button Down Collar Formal Shirt", "rank": 3, "price": 758.0, "rating": 4.2, "in_stock": true}, {"pid": "SHTFU73NHNAFBCCZ", "why": "White Slim Fit Solid Spread Collar Formal Shirt, in stock, lowest price Rs 664 but lower rating 3.7/5", "title": "Men Slim Fit Solid Spread Collar Formal Shirt", "rank": 5, "price": 664.0, "rating": 3.7, "in_stock": true}], "summary": "The search returned three white slim‑fit formal shirts that meet all your criteria. The top pick offers the best rating and a good price, while the alternatives are cheaper or have slightly higher ratings to consider."}
```

## Q2: women floral print shirt

**baseline** (ok, best = SHTFTGV4ATUQYVWY, relevant = 1.0, invented PIDs = 0)

```
- Best Product: SHTFTGV4ATUQYVWY Women Regular Fit Floral Print Button Down Collar Casual Shirt  
- Why: This shirt offers a classic button‑down style that’s versatile for both casual and semi‑formal occasions. It has a regular fit that balances comfort and a polished look, and the floral print is vibrant yet not overly busy. The button‑down collar provides a tidy neckline that pairs well with blazers or sweaters.  
- Alternative (optional): SHTF4FWJ6UQAHRQM Women Regular Fit Floral Print Casual Shirt – a similar regular fit but with a simpler collar for a more relaxed vibe.
```

**improved** (ok, best = SHTFDYMYV7AUKE9R, relevant = 1.0, invented PIDs = 0)

```
{"status": "ok", "best": {"pid": "SHTFDYMYV7AUKE9R", "why": "Highest rating of 4.5/5 among in‑stock floral shirts, priced at Rs 799 with a 46% discount", "title": "Women Slim Fit Floral Print Button Down Collar Casual Shirt", "rank": 7, "price": 799.0, "rating": 4.5, "in_stock": true}, "alternatives": [{"pid": "SHTFJZC45ZFYGSYH", "why": "Good rating of 4.3/5, lower price Rs 635 (51% off) – cheaper option", "title": "Women Slim Fit Floral Print Casual Shirt", "rank": 2, "price": 635.0, "rating": 4.3, "in_stock": true}, {"pid": "SHTF4FWJ6UQAHRQM", "why": "Solid rating of 4.1/5, slightly higher price Rs 790 (50% off) – better color variety", "title": "Women Regular Fit Floral Print Casual Shirt", "rank": 1, "price": 790.0, "rating": 4.1, "in_stock": true}], "summary": "The search returned several in‑stock women’s floral print shirts. The top pick offers the best rating, while alternatives provide cheaper or slightly different style options."}
```

## Q3: women black track pants with zip pockets

**baseline** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
There are no good products that fit the request based on the retrieved results.
```

**improved** (ok, best = TKPFWGMH448EGQTF, relevant = 1.0, invented PIDs = 0)

```
{"status": "ok", "best": {"pid": "TKPFWGMH448EGQTF", "why": "Solid Women Black Track Pants from tom burg with a 58% discount, priced at Rs 499, and a 4.3/5 rating. It includes side zip pockets for secure storage.", "title": "Solid Women Black Track Pants", "rank": 1, "price": 499.0, "rating": 4.3, "in_stock": true}, "alternatives": [{"pid": "TKPFKFK4ESGQ95CK", "why": "Solid Women Black Track Pants with side zip pockets, priced at Rs 765, 3.7/5 rating – slightly higher price but still a good value.", "title": "Solid Women Black Track Pants", "rank": 3, "price": 765.0, "rating": 3.7, "in_stock": true}, {"pid": "TKPFKFZFXZT5XQYZ", "why": "Solid Women Black, Grey Track Pants with side zip pockets, priced at Rs 740, 3.7/5 rating – cheaper than the best but offers a dual‑color option.", "title": "Solid Women Black, Grey Track Pants", "rank": 7, "price": 740.0, "rating": 3.7, "in_stock": true}], "summary": "The search returned several black track pants with zip pockets. The top pick is the most affordable with a solid rating, while the alternatives offer slightly higher prices or a different color option."}
```

## Q4: women pure cotton t-shirt pack of 3

**baseline** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
There are no good products that fit the request based on the retrieved results.
```

**improved** (ok, best = TSHFVYHB7YTVJ77M, relevant = 1.0, invented PIDs = 0)

```
{"status": "ok", "best": {"pid": "TSHFVYHB7YTVJ77M", "why": "Solid Women Round Neck Multicolor T-Shirt from Sevnix, 4.3/5 rating, Rs 489, pure cotton, pack of 3, in stock", "title": "Solid Women Round Neck Multicolor T-Shirt (Pack of 3)", "rank": 6, "price": 489.0, "rating": 4.3, "in_stock": true}, "alternatives": [{"pid": "TSHFWBWECMBZ9D8A", "why": "Solid Women Round Neck Black T-Shirt from Scott International, 4.1/5 rating, Rs 499, pure cotton, pack of 3, in stock", "title": "Solid Women Round Neck Black T-Shirt (Pack of 3)", "rank": 1, "price": 499.0, "rating": 4.1, "in_stock": true}], "summary": "The results include several women’s pure cotton t-shirts in packs of three. The top pick is a higher‑rated Sevnix option at Rs 489, with a solid alternative from Scott International at Rs 499."}
```

## Q5: black running shoes for men

**baseline** (ok, best = SHOFTZFPNCU4VWHK, relevant = 1.0, invented PIDs = 0)

```
- Best Product: SHOFTZFPNCU4VWHK Running Shoes For Men (Black)  
- Why: This product is a dedicated black running shoe for men, matching the exact color and gender requested. It is the simplest, most direct choice among the available options, ensuring you get a shoe that is fully black and designed specifically for male runners.  
- Alternative (optional): SHOFTZFQP8W9G86F Running Shoes For Men (Black)
```

**improved** (ok, best = SHOFU9EEZG46B55M, relevant = 1.0, invented PIDs = 0)

```
{"status": "ok", "best": {"pid": "SHOFU9EEZG46B55M", "why": "Highest rating of 4.4/5 among black running shoes for men, in stock, priced at Rs 675 with a 9% discount.", "title": "VX-7.0 Running Shoes For Men (Maroon, Black)", "rank": 6, "price": 675.0, "rating": 4.4, "in_stock": true}, "alternatives": [{"pid": "SHOFTZFPNCU4VWHK", "why": "Good rating of 4.1/5, cheaper at Rs 379 with a 62% discount.", "title": "Running Shoes For Men (Black)", "rank": 1, "price": 379.0, "rating": 4.1, "in_stock": true}, {"pid": "SHOF3A22BBDQQY8F", "why": "Same rating of 4.1/5 but higher price of Rs 999 with a 31% discount.", "title": "Running Shoes For Men (Black, Grey)", "rank": 3, "price": 999.0, "rating": 4.1, "in_stock": true}], "summary": "The search returned several black running shoes for men; the top pick is the best‑rated option at a reasonable price, with two alternatives that offer either lower cost or a comparable rating at a higher price. Check the price, discount, and material details to match your preferences."}
```

## V1: women full sleeve sweatshirt cotton

**baseline** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
"There are no good products that fit the request based on the retrieved results."
```

**improved** (ok, best = SWSFVZRFS7GHGKSF, relevant = nan, invented PIDs = 0)

```
{"status": "ok", "best": {"pid": "SWSFVZRFS7GHGKSF", "why": "Full Sleeve Solid Women Sweatshirt in Pure Cotton, Yellow, 4.0/5 rating, in stock, Rs 799", "title": "Full Sleeve Solid Women Sweatshirt", "rank": 1, "price": 799.0, "rating": 4.0, "in_stock": true}, "alternatives": [], "summary": "The search returned one product that matches all your criteria: a pure cotton full‑sleeve sweatshirt for women. It’s in stock, reasonably priced, and has a solid rating."}
```

## V2: men slim jeans blue

**baseline** (ok, best = JEAFESND4QWQUBZD, relevant = nan, invented PIDs = 0)

```
- Best Product: JEAFESND4QWQUBZD Slim Men Blue Jeans  
- Why: This item is specifically labeled “Blue Jeans,” matching the user’s request for men’s slim blue jeans. It’s the only product with a plain blue color (not dark blue), making it the most accurate fit.  
- Alternative: JEAFJM2P2FQQJF3C Slim Men Dark Blue Jeans (if a slightly darker shade is acceptable).
```

**improved** (ok, best = JEAFESND4QWQUBZD, relevant = nan, invented PIDs = 0)

```
{"status": "ok", "best": {"pid": "JEAFESND4QWQUBZD", "why": "It is a Slim Men Blue Jeans with the highest rating of 5.0/5 and is in stock.", "title": "Slim Men Blue Jeans", "rank": 1, "price": 1319.0, "rating": 5.0, "in_stock": true}, "alternatives": [{"pid": "JEAFEKQZWYPVGYCJ", "why": "Same fit and colour, slightly lower rating (4.3/5) but cheaper at Rs 1,199.", "title": "Slim Men Blue Jeans", "rank": 4, "price": 1199.0, "rating": 4.3, "in_stock": true}, {"pid": "JEAFDWG9K7KPKHS8", "why": "Also a Slim Men Blue Jeans, but much cheaper at Rs 599 with a rating of 4.1/5.", "title": "Slim Men Blue Jeans", "rank": 5, "price": 599.0, "rating": 4.1, "in_stock": true}], "summary": "The search returned several blue slim jeans for men. The top pick has the best rating, while the alternatives offer lower prices or slightly lower ratings for similar style."}
```

## T1: black runing shoes for men

**baseline** (ok, best = SHOFTZFPNCU4VWHK, relevant = 1.0, invented PIDs = 0)

```
- Best Product: **SHOFTZFPNCU4VWHK** Running Shoes For Men (Black)  
- Why: This shoe is specifically listed in black only, matching the user’s preference for a single-color black running shoe for men. The listing is straightforward with no additional colors, ensuring a clear fit for the requested style.  
- Alternative (optional): **SHOFTZFQP8W9G86F** Running Shoes For Men (Black) – another black-only option that would also meet the same requirement.
```

**improved** (ok, best = SHOFU9EEZG46B55M, relevant = 1.0, invented PIDs = 0)

```
{"status": "ok", "best": {"pid": "SHOFU9EEZG46B55M", "why": "Highest rating (4.4/5) among black running shoes for men, in stock, price Rs 675 with 9% discount", "title": "VX-7.0 Running Shoes For Men (Maroon, Black)", "rank": 6, "price": 675.0, "rating": 4.4, "in_stock": true}, "alternatives": [{"pid": "SHOFTZFPNCU4VWHK", "why": "Cheapest option at Rs 379 with 62% off, still in stock but lower rating (4.1/5)", "title": "Running Shoes For Men (Black)", "rank": 1, "price": 379.0, "rating": 4.1, "in_stock": true}, {"pid": "SHOF3A22BBDQQY8F", "why": "Higher rating (4.1/5) than many options, price Rs 999 with 31% off", "title": "Running Shoes For Men (Black, Grey)", "rank": 3, "price": 999.0, "rating": 4.1, "in_stock": true}], "summary": "The search returned six suitable black running shoes for men. The top pick is the vector VX‑7.0 (PID SHOFU9EEZG46B55M) for its best rating, while alternatives offer either lower cost or slightly better value."}
```

## A1: men leather wallet

**baseline** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
There are no good products that fit the request based on the retrieved results.
```

**improved** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
{"status": "no_good_products", "best": null, "alternatives": [], "summary": "The search returned a pair of suspenders, not a leather wallet. No product in the results matches the requested type, gender, or material, so there are no suitable options to recommend."}
```

## A2: women high heels

**baseline** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
"There are no good products that fit the request based on the retrieved results."
```

**improved** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
{"status": "no_good_products", "best": null, "alternatives": [], "summary": "The search results contain only a pack of no‑show socks, which does not match the user’s request for women’s high heels. No product in the list meets the required type, so no suitable items can be recommended."}
```

## A3: kids school bag

**baseline** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
"There are no good products that fit the request based on the retrieved results."
```

**improved** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
{"status": "no_good_products", "best": null, "alternatives": [], "summary": "The search results do not contain any kids school bags. The shop currently offers only raincoats and men’s footwear, so you may need to refine your search or check other categories for a suitable school bag."}
```

## A4: wedding saree

**baseline** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
"There are no good products that fit the request based on the retrieved results."
```

**improved** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
{"status": "no_good_products", "best": null, "alternatives": [], "summary": "The search results do not contain any wedding sarees; all items listed are blazers or a beanie. The shop currently has no product that matches the requested type, so you may want to refine your search or check back later for saree options."}
```

## A5: laptop backpack

**baseline** (no_good_products, best = nan, relevant = nan, invented PIDs = 0)

```
"There are no good products that fit the request based on the retrieved results."
```

**improved** (no_results, best = nan, relevant = nan, invented PIDs = 0)

```
{"status": "no_results", "best": null, "alternatives": [], "summary": "No product of our shop matches \"laptop backpack\". Try other words or fewer filters."}
```

## O1: gift for my father

**baseline** (ok, best = CTPFVQSGZGPHUEFX, relevant = nan, invented PIDs = 0)

```
- Best Product: CTPFVQSGZGPHUEFX nu-Lite Satin Tie & Cufflink (Black)  
- Why: A black tie and cufflink set is a timeless, versatile gift that suits formal occasions and everyday wear. It offers a polished look with minimal maintenance, making it perfect for a father who appreciates classic style and quality.  
- Alternative: CTPFVZHSA7G4PFC5 nu-Lite Satin Tie & Cufflink (Grey)
```

**improved** (ok, best = CTPFVZHSA7G4PFC5, relevant = nan, invented PIDs = 0)

```
{"status": "ok", "best": {"pid": "CTPFVZHSA7G4PFC5", "why": "In stock, highest rating 4.3/5, reasonable price Rs 499 after 80% off", "title": "nu-Lite Satin Tie & Cufflink (Grey)", "rank": 1, "price": 499.0, "rating": 4.3, "in_stock": true}, "alternatives": [{"pid": "CTPFVQSGZGPHUEFX", "why": "In stock, decent rating 3.3/5, same price Rs 499", "title": "nu-Lite Satin Tie & Cufflink (Black)", "rank": 3, "price": 499.0, "rating": 3.3, "in_stock": true}, {"pid": "CTPFVPH2JGZRW6FV", "why": "In stock, lower rating 2.7/5, same price Rs 499", "title": "nu-Lite Satin Tie & Cufflink (Pink)", "rank": 4, "price": 499.0, "rating": 2.7, "in_stock": true}], "summary": "The search returned several tie options suitable as a gift for a father. The top pick is the grey satin tie with the best rating, and two other ties are offered as alternatives with slightly lower ratings but identical pricing."}
```

## O2: comfortable shoes for walking

**baseline** (ok, best = SHOFZFF2NYHQGQ2K, relevant = nan, invented PIDs = 0)

```
- Best Product: SHOFZFF2NYHQGQ2K Walking Shoes For Men (White)  
- Why: It is specifically labeled as walking shoes, designed for comfort during walking, and offers a classic white color that is versatile and suitable for many occasions.  
- Alternative (optional): SHOFM8GQ2YR6Q5FA Walking Shoes For Women (Blue)  
```

**improved** (ok, best = SHOFM8GQ2YR6Q5FA, relevant = nan, invented PIDs = 0)

```
{"status": "ok", "best": {"pid": "SHOFM8GQ2YR6Q5FA", "why": "Walking shoes for women, 3.6/5 rating, Rs 474, in stock – meets the user’s request for comfortable walking shoes.", "title": "Walking Shoes For Women (Blue)", "rank": 3, "price": 474.0, "rating": 3.6, "in_stock": true}, "alternatives": [{"pid": "SHOFM8GQWNZDGS4Q", "why": "Walking shoes for men, same price and rating (3.6/5), also in stock – a good alternative if the shopper prefers a different colour or gender.", "title": "Walking Shoes For Men (Grey)", "rank": 4, "price": 474.0, "rating": 3.6, "in_stock": true}], "summary": "The search returned two top‑rated walking shoes that fit the comfort requirement. Both are priced at Rs 474 and are in stock; choose the one that matches your preferred gender or colour."}
```

## O3: cheap cotton shirt for men under 500

**baseline** (ok, best = SHTF4EHG2KU74RX5, relevant = nan, invented PIDs = 0)

```
- Best Product: SHTF4EHG2KU74RX5 Men Slim Fit Solid Mandarin Collar Casual Shirt  
- Why: This shirt is made from cotton, fits the slim‑fit style, and is listed at a price that is below ₹500, making it the most economical choice for a cheap cotton shirt for men.  
- Alternative (optional): SHTF7YU8UYN399HS Men Slim Fit Solid Mandarin Collar Party Shirt – also a cotton shirt under ₹500.
```

**improved** (ok, best = SHTF4EHG2KU74RX5, relevant = nan, invented PIDs = 0)

```
{"status": "ok", "best": {"pid": "SHTF4EHG2KU74RX5", "why": "Rs 398, highest discount (77%) and lowest price among in‑stock men’s cotton‑blend shirts", "title": "Men Slim Fit Solid Mandarin Collar Casual Shirt", "rank": 1, "price": 398.0, "rating": 3.7, "in_stock": true}, "alternatives": [{"pid": "SHTFAMHZ5JPDBGH4", "why": "Rs 411, slightly higher price but same fit and rating 3.5/5", "title": "Men Slim Fit Solid Casual Shirt", "rank": 15, "price": 411.0, "rating": 3.5, "in_stock": true}, {"pid": "SHTF7YU8UYN399HS", "why": "Rs 419, same fit and rating 3.7/5, slightly higher price", "title": "Men Slim Fit Solid Mandarin Collar Party Shirt", "rank": 13, "price": 419.0, "rating": 3.7, "in_stock": true}], "summary": "The search returned seven in‑stock men’s cotton‑blend shirts priced under Rs 500. The top pick is the Rs 398 shirt with the best discount, and two alternatives are offered at Rs 411 and Rs 419 for slightly higher quality or rating."}
```

