# Caption Object Change Decomposition

## Inputs

- base: `res-opd/eval_results/instruct/full/Qwen3VL-8B-Instruct/train5000_test1000_original_sr1p0/eval_results.jsonl`
- test_json: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/data/test_1000.json`

## Aggregate Metrics

| run | records | CHAIRi | CHAIRs proxy | ObjPrec | ObjRec | ObjF1 | mentioned | halluc | correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 1000 | 0.3025 | 0.5690 | 0.6975 | 0.7411 | 0.7186 | 3074 | 930 | 2144 |
| rkl_step312 | 1000 | 0.2894 | 0.5590 | 0.7106 | 0.7461 | 0.7279 | 3027 | 876 | 2151 |
| riskmask_p30_step100 | 1000 | 0.2873 | 0.5480 | 0.7127 | 0.7483 | 0.7301 | 3021 | 868 | 2153 |

## Paired Change Totals

| run | removed halluc | added halluc | net halluc | removed correct | added correct | net correct | net mentioned |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rkl_step312 | 342 | 288 | -54 | 136 | 143 | 7 | -47 |
| riskmask_p30_step100 | 323 | 261 | -62 | 131 | 140 | 9 | -53 |

## rkl_step312

### Category Decomposition

| category | removed halluc | share removed halluc | added halluc | net halluc | removed correct | share removed correct | added correct | net correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uncertain_small_background | 197 | 0.5760 | 165 | -32 | 83 | 0.6103 | 77 | -6 |
| ordinary_visual_object | 67 | 0.1959 | 60 | -7 | 23 | 0.1691 | 31 | 8 |
| text_logo_graphic | 64 | 0.1871 | 53 | -11 | 30 | 0.2206 | 34 | 4 |
| color_attribute_as_object | 14 | 0.0409 | 10 | -4 | 0 | 0.0000 | 1 | 1 |

### Flag Decomposition

| flag | removed halluc | added halluc | net halluc | removed correct | added correct | net correct |
| --- | --- | --- | --- | --- | --- | --- |
| text_or_graphic | 68 | 55 | -13 | 30 | 34 | 4 |
| color_attribute | 14 | 10 | -4 | 0 | 1 | 1 |
| uncertain_small_background | 254 | 211 | -43 | 107 | 103 | -4 |
| uppercase_mention | 4 | 2 | -2 | 3 | 2 | -1 |

### Removed Hallucinated: Text/Logo/Graphic

| image | object | category | group | terms | context |
| --- | --- | --- | --- | --- | --- |
| 84270 | airplane | text_logo_graphic | vehicle | airplane | ...bers include "219-224" and "225-232". The signs also feature small icons, including a red airplane symbol and a blue airplane symbol. To the left, a green sign with a white "E" (likely ind... |
| 161609 | airplane | text_logo_graphic | vehicle | airplane | ...n is visible. It reads “Millbrae” in large, white, sans-serif letters, with a small white airplane icon and the letters “BART” to the right of it. To the right of “Millbrae,” there is a wh... |
| 468965 | airplane | text_logo_graphic | vehicle | airplane, airplanes | ...** The central focus is on four children engaged in playful activity with handmade paper airplanes. - **Boy in the Center (Back to Camera):** A boy with short black hair, wearing a light ... |
| 76416 | apple | text_logo_graphic | food | Apple, apple | ...hield. - The entire side of the bus is covered in a large, colorful advertisement for the Apple iPod. - The ad features a repeating geometric pattern of stylized, overlapping shapes in ... |
| 432085 | apple | text_logo_graphic | food | apple | ...t. - A box of tissues with a colorful, cartoonish design (featuring what looks like a red apple and a yellow sun) is placed behind the water bottle. - A small, clear plastic cup is part... |
| 479596 | bear | text_logo_graphic | animal | bear | ...eft and central portions of the frame. The oranges have a glossy, textured peel, and many bear small, circular blue and white stickers, which appear to be brand or quality labels. The ... |
| 170595 | bed | text_logo_graphic | furniture | bed | ...l impression is one of a simple, satisfying, and comforting breakfast, perhaps enjoyed in bed on a lazy morning. There is no visible text or branding in the image. |
| 163117 | bird | text_logo_graphic | animal | bird | ...lf of the frame. They are all identical in design: a stylized, angular shape resembling a bird or a geometric symbol, with a red body and black accents. Their bright red color contrast... |
| 360097 | bird | text_logo_graphic | animal | bird | ...om right corner, there is a faint, stylized black signature or watermark that resembles a bird or a cursive letter “S”. - The overall lighting is even and diffuse, suggesting an overca... |
| 405691 | bird | text_logo_graphic | animal | bird, rooster | ...mpty glass. Its label is colorful and features a cartoonish illustration of a red bird or rooster. The text on the label is partially legible and includes the words “BENJY” and “MATRAI,” ... |
| 360097 | boat | text_logo_graphic | vehicle | ferry, ship | ...is a high-angle, top-down photograph of a concrete dock or pier area, likely at a port or ferry terminal. The scene is dominated by piles of luggage and a prominent white railing. **Fo... |
| 369503 | boat | text_logo_graphic | vehicle | sailboat | ... with a white horizontal stripe near the hem, which features a small red star and a white sailboat logo. Their left arm is bent, and their hand is near their neck or shoulder. Their dark p... |
| 54654 | book | text_logo_graphic | indoor_other | book | ... the words "my raggedy" and "don't" can be discerned, suggesting it might be a children's book or poster. To the right of the person, a white door or cabinet is visible, with several ... |
| 119365 | book | text_logo_graphic | indoor_other | book | ...e center, a poster with a blue border and a speech bubble graphic shows a child reading a book. - To the right, a prominent white poster with a blue border is titled "Building vocabula... |
| 22892 | bowl | text_logo_graphic | kitchen_tableware | container | ...e a young tomato or pepper plant, with several stems and leaves. It is planted in a small container filled with dark soil. The pot is partially visible, showing a yellow and white label wit... |
| 194471 | bowl | text_logo_graphic | kitchen_tableware | bowl | ...ment of athletic motion. He is riding a skateboard on the curved edge of a concrete skate bowl. - He is wearing a black short-sleeved t-shirt with a white graphic on the chest, light-w... |
| 28993 | car | text_logo_graphic | vehicle | cars | ...ue to the depth of field. To the left, a street runs parallel to the sidewalk, with a few cars and street signs visible. A blue and white vertical sign is mounted on a pole on the left... |
| 109900 | car | text_logo_graphic | vehicle | SUV | ... in discussion or waiting. - **Vehicle:** To the right, a white, rugged, four-wheel-drive SUV is parked. It has a black roof rack and a tall antenna on the front. The vehicle is clear... |
| 143572 | car | text_logo_graphic | vehicle | car | ...printed in large, white, sans-serif letters. - To the left, the logo and text “esurance car insurance” are visible. - To the right, the URL “usopen.org” is displayed in white. - A... |
| 336628 | car | text_logo_graphic | vehicle | Car, car | ...an in a highly stylized and colorful costume standing on the open-air platform of a cable car, likely during a parade or festive street event. **The Central Figure:** The man is the ... |
| 460929 | car | text_logo_graphic | vehicle | car | This is a candid, sunlit photograph of a casual meal, likely taken from inside a car, as suggested by the window in the background. The scene is composed of a hot dog and a b... |
| 468965 | car | text_logo_graphic | vehicle | car | ... white design with a large red letter “E” and cartoon-like illustrations, including a red car and a blue bird. - **Child in the Foreground (Left Side):** In the lower-left foreground... |
| 575081 | car | text_logo_graphic | vehicle | cars | ...d brightly patterned pajama pants with a red, blue, and white design featuring cartoonish cars and road signs. His bare feet are on the wooden floor, and he is wearing white socks. His... |
| 223747 | cell phone | text_logo_graphic | electronics_appliance | phone | ...w, with visible pixelation and blurriness, suggesting it was taken with a small camera or phone in a low-light environment. There is no text or discernible fine detail beyond the basic ... |
| 356612 | cell phone | text_logo_graphic | electronics_appliance | phone | ...on its front grille. - The yellow bus has text on its side, including "THE COLLEGE" and a phone number. This image is a vivid snapshot of a moment in time, blending the mundane with th... |

### Removed Hallucinated: Uncertain/Small/Background

| image | object | category | group | terms | context |
| --- | --- | --- | --- | --- | --- |
| 110884 | airplane | uncertain_small_background | vehicle | airplane | ...s is a photograph of a compact, utilitarian bathroom, likely from a hotel room, train, or airplane cabin, characterized by its small size and functional design. The image has a strong gree... |
| 56344 | apple | uncertain_small_background | food | Apple | ... is a slim, white, full-sized keyboard with a standard QWERTY layout. It appears to be an Apple Magic Keyboard. - To the right of the keyboard is a white, ergonomic computer mouse, like... |
| 425390 | apple | uncertain_small_background | food | Apple | ...ambient light. **The Laptop** - The cat is resting on a silver-colored laptop, likely an Apple MacBook, given the design of the trackpad and the overall aesthetic. - The laptop is open... |
| 210388 | backpack | uncertain_small_background | accessory | backpack | ...N” and possibly “SUMMIT” below it. He is also wearing yellow gloves and has a small, dark backpack or bag slung over his shoulder. * **On the right:** A woman with shoulder-length brown ... |
| 57150 | bear | uncertain_small_background | animal | bear | ...laid skirt, white tights, and dark shoes. She is holding a very large, fluffy white teddy bear that towers over her. The bear is oversized, with a prominent red bow tied around its nec... |
| 231879 | bear | uncertain_small_background | animal | bear | ...entle, focused expression. The cake is decorated with a small, brown figurine (possibly a bear or animal) on top and a single, thin candle. **Background Elements:** - Behind the main ... |
| 315492 | bear | uncertain_small_background | animal | Bear, bear | ...nded toward it, suggesting she is either pulling a sheet or adjusting the roll. - **Teddy Bear:** On the floor, to the child’s left, sits a large, brown teddy bear. The bear is lying o... |
| 370478 | bear | uncertain_small_background | animal | bears | ...rter or sheet featuring a repeating pattern of small, dark cartoon-like figures (possibly bears or animals) and stars. - A black jacket or piece of clothing is draped over the edge of t... |
| 188439 | bed | uncertain_small_background | furniture | bed | ...arcane. The cane is stacked high, extending far beyond the cab and the sides of the truck bed, forming a dense, brownish-green mass. The sugarcane stalks are bundled together and secu... |
| 368038 | bed | uncertain_small_background | furniture | bed | ...green tree. **Environment and Background** * **Ground:** The cabooses sit on a gravel bed next to a patch of grassy, slightly uneven ground. The grass is a mix of green and brown,... |
| 550471 | bed | uncertain_small_background | furniture | bed | ... pale, almost translucent rind. The slice is positioned slightly off-center, resting on a bed of delicate green microgreens. - The microgreens, which appear to be radish sprouts or a ... |
| 575081 | bed | uncertain_small_background | furniture | bed | ...re strewn across the floor, connecting the gaming equipment. A small, woven basket or pet bed is on the floor near the boy’s feet. A small, dark-colored mug sits on a shelf on the ent... |
| 69213 | bench | uncertain_small_background | outdoor_sign | bench | ...le:** In the background, other people can be seen. One person is sitting on a low step or bench near the building, wearing a red top. Another person is partially visible further back on... |
| 87038 | bench | uncertain_small_background | outdoor_sign | bench | ...nd is riding a small, black-and-white BMX bike. The bike is positioned near a low, wooden bench or platform. - The ground is a flat, gray asphalt surface, typical of a skatepark. Severa... |
| 386352 | bench | uncertain_small_background | outdoor_sign | bench | ... the court area from the spectator seating. - Behind the railing, a blue plastic chair or bench is partially visible. - The setting appears to be an outdoor or well-lit indoor tennis co... |
| 411530 | bench | uncertain_small_background | outdoor_sign | bench | ...k, and gray—and are waiting to board the lift. Some are standing, others are sitting on a bench near a small building. **Midground:** - A chairlift system is the central focus. It cons... |
| 431896 | bench | uncertain_small_background | outdoor_sign | benches | ... warm glow. Further down the platform, other station infrastructure is visible, including benches and what appears to be a ticket machine or information kiosk. - **Spatial Relationship:**... |
| 476491 | bicycle | uncertain_small_background | vehicle | bike | ...oreground is a dark, textured asphalt or concrete surface, likely a pedestrian walkway or bike lane. It is marked with large, bold yellow Japanese characters: “スクールゾーン” (Sukūru Zōn), w... |
| 534827 | bicycle | uncertain_small_background | vehicle | bike, bikes | ... strong backlighting creates silhouettes and highlights the edges of the riders and their bikes. - Long, dark shadows are cast by the motorcycles and riders onto the road, stretching to... |
| 270122 | bird | uncertain_small_background | animal | birds | ... cool, moody atmosphere. - A few small, dark specks are visible high in the sky, possibly birds or distant objects. - The ocean extends to the horizon, appearing vast and unbroken. **T... |
| 400922 | bird | uncertain_small_background | animal | bird | ...is a vibrant blue, filled with large, fluffy white clouds. A small, dark object, likely a bird or a drone, is captured in mid-flight against the sky, positioned above the KeyBank sign.... |
| 540932 | boat | uncertain_small_background | vehicle | sailboard | ...submerged or resting on the sand. It appears to be a piece of equipment, possibly a small sailboard or a kiteboard, with a black and yellow board attached to it. A person is sitting on the ... |
| 78426 | book | uncertain_small_background | indoor_other | book | ...stands out among them. - **In the bottom left corner:** The edge of a notebook or another book is partially visible. **Background and Environment:** - The background is softly blurred... |
| 415194 | book | uncertain_small_background | indoor_other | book | ... a few small items, including a brown box and a small, rectangular object that might be a book or a box of matches. - **Window:** On the far right, a window with vertical blinds is par... |
| 21167 | bottle | uncertain_small_background | kitchen_tableware | bottles | ...eading to another room. Through the doorway, a shelving unit with various items (possibly bottles or decorative objects) can be seen. - To the right of the man, there is a white door fram... |

### Removed Correct: Text/Logo/Graphic

| image | object | category | group | terms | context |
| --- | --- | --- | --- | --- | --- |
| 185599 | apple | text_logo_graphic | food | apple, apples | ...red. - **Center (Background):** Nestled between the two foreground fruits is a cluster of apples. One apple is visible directly behind the left orange, showing a blend of red and yellow ... |
| 481390 | bench | text_logo_graphic | outdoor_sign | bench | ... on the sideline, including a man in a black jacket and another in a white shirt near the bench area. **The Players and Their Actions:** There are eight players clearly visible on the ... |
| 384527 | book | text_logo_graphic | indoor_other | book, books | ...pears to be a classic model. - Behind the green chairs, a wooden bookshelf is filled with books. The spines of the books are mostly red and white, creating a striking visual contrast. -... |
| 473219 | book | text_logo_graphic | indoor_other | book | ...ny. He is wearing a dark suit, a white shirt, and a patterned tie. He holds an open black book, likely a Bible or wedding service book, in his hands and appears to be reading or speaki... |
| 396274 | broccoli | text_logo_graphic | food | broccoli | ...robust plants with broad, dark green, slightly crinkled leaves. These are identifiable as broccoli plants, with some showing developing green florets. The leaves have prominent, lighter-co... |
| 86483 | car | text_logo_graphic | vehicle | car | ...e of the sign on the left. Further in the background, more trees and a glimpse of a white car parked on a street can be seen. - **Foreground:** The sign sits on a low, stone-faced wal... |
| 130386 | car | text_logo_graphic | vehicle | SUV | ...ed on a sidewalk. In the background, a street with traffic is visible, including a silver SUV and a red double-decker bus partially seen on the left. Several pedestrians can be seen w... |
| 507667 | car | text_logo_graphic | vehicle | SUV, van | ...her person in a pink shirt is partially visible. - **Further Background:** A white van or SUV is parked behind the motorcycle. Further back, there are trees, other pedestrians, and wh... |
| 481390 | chair | text_logo_graphic | furniture | seats | ...sible on the red sideline barrier. * The “Coca-Cola” logo is visible on the black bench seats in the background. **Overall Composition:** The photograph captures a moment of intense ... |
| 500478 | chair | text_logo_graphic | furniture | seats | ...is a vibrant green. - **Background:** Behind the pitcher, there are rows of green stadium seats. The seats are mostly empty, but a few spectators are visible. They are seated in the sta... |
| 189752 | cup | text_logo_graphic | kitchen_tableware | cup | ... long-sleeved shirt and appear to be eating or drinking from a straw. A small white paper cup with red and black text is on the table in front of them. The text on the cup is partiall... |
| 555009 | cup | text_logo_graphic | kitchen_tableware | Cup, cup | ... the keyboard. The headphones are not on anyone's head. * **Other Objects:** * **Cup:** A green and white patterned ceramic mug or cup sits on the desk between the two monito... |
| 157418 | dining table | text_logo_graphic | furniture | dining table, table | This is a close-up, slightly angled photograph of a casual meal setup, likely on a dining table. The scene is composed of several food items and condiments, with a focus on a can of SPA... |
| 160012 | dining table | text_logo_graphic | furniture | table | ... board, with another pizza partially visible in the background. A person is seated at the table, interacting with the food. **Foreground and Center:** - The main subject is a large, ro... |
| 277689 | dining table | text_logo_graphic | furniture | Table, table | This is a vibrant, sun-drenched photograph of an outdoor wedding or celebration cake table, set against a tropical backdrop. The composition is carefully arranged to draw the eye t... |
| 261888 | horse | text_logo_graphic | animal | horse, horses | ...a wide-angle, eye-level photograph capturing a cyclist paused on a paved road, facing two horses that are walking toward him in the distance. The scene is set on a bright, sunny day unde... |
| 492362 | hot dog | text_logo_graphic | food | hot dogs | ...od items. Visible text includes "HALAL FOOD" in green and yellow, and below it, images of hot dogs and a bowl of what appears to be beans or nuts. A green sign at the bottom reads "Salsa, ... |
| 35326 | oven | text_logo_graphic | electronics_appliance | Stove, oven, stove | ...ash area, captured from a slightly elevated, eye-level perspective. **Central Focus: The Stove and Countertop** The main subject is a black, flat electric cooktop with four circular he... |
| 283717 | oven | text_logo_graphic | electronics_appliance | oven | ...tral Focus: The Appliance Stack** - **Top Appliance**: A vintage, silver-colored toaster oven. It has a heavily used, reflective surface with visible smudges and darkened areas, parti... |
| 330818 | oven | text_logo_graphic | electronics_appliance | oven, stove, stovetop | ...nate, brass-colored handles and knobs, giving it a classic, somewhat vintage look. On the stovetop, there are several pans, including a large frying pan on the left side, and a small pot o... |
| 93437 | person | text_logo_graphic | person | Man, man | This is a candid, indoor photograph of an elderly man, likely in his 70s or 80s, who is the central focus of the image. He is smiling broadly a... |
| 188439 | person | text_logo_graphic | person | People, person | ...helter. - A small, white sign with red markings is mounted on a post near the pole. - **People:** A person wearing a blue shirt and dark pants is visible standing in the background, ne... |
| 251824 | person | text_logo_graphic | person | person | This is a close-up, first-person perspective photograph of a person’s hands working on a craft project on a wooden desk. T... |
| 330818 | person | text_logo_graphic | person | man, person | This is a candid, slightly grainy photograph of a person working in a commercial kitchen, likely a diner or small restaurant, captured from a slig... |
| 369812 | person | text_logo_graphic | person | people | ...ble-decker bus is visible, a classic symbol of London. - Other cars and a bus stop with people waiting can be seen in the distance. - **Sidewalk and Pedestrians:** A paved sidewalk ru... |

### Removed Correct: Uncertain/Small/Background

| image | object | category | group | terms | context |
| --- | --- | --- | --- | --- | --- |
| 86755 | backpack | uncertain_small_background | accessory | backpack | ...ark jacket with a red or magenta panel on the shoulders and chest, dark pants, and a dark backpack. They are holding ski poles and have a pair of red skis on their feet. - The second skier... |
| 309964 | backpack | uncertain_small_background | accessory | Backpack, backpack | ...dle. It appears to be filled with small, greenish items, possibly fish or bait. * **Backpack:** Between the two people, on the ground, is a colorful, pixelated or mosaic-patterned ba... |
| 134722 | bench | uncertain_small_background | outdoor_sign | bench | ...f with a red trim. A red downpipe runs vertically down the side of the brick structure. A bench is visible on the platform near this building. **Background and Environment:** Behind th... |
| 87038 | bicycle | uncertain_small_background | vehicle | bike | ...iddle of a flip or kickflip. - To the left of the airborne skater, another rider on a BMX bike is visible. This person is wearing a pink or magenta hooded sweatshirt and dark pants, an... |
| 459809 | boat | uncertain_small_background | vehicle | sailboats, vessels | ...ndmass or mountain range forms the horizon line. A few small, indistinct shapes, possibly sailboats or other vessels, are visible on the water near the left edge of the frame. **Foreground... |
| 222825 | bottle | uncertain_small_background | kitchen_tableware | bottle | ...ount sink is set into the countertop, with a white, curved faucet. A small, white plastic bottle (possibly dish soap) and a black smartphone are resting on the counter near the sink. - *... |
| 349480 | bottle | uncertain_small_background | kitchen_tableware | bottle | ..., a red apple, and a cluster of purple grapes. * **Beverages:** A small, dark green bottle (possibly wine or soda) and a small, white ceramic pitcher with a blue floral pattern. ... |
| 498747 | bottle | uncertain_small_background | kitchen_tableware | bottle | ...ing motion and the dynamic nature of the party. The hand appears to be holding a glass or bottle, which is partially visible and blurred. **Foreground and Background Details:** - In the... |
| 425226 | bowl | uncertain_small_background | kitchen_tableware | container | ...hair. Their posture suggests they are looking for something specific, perhaps a bottle or container, located on a higher shelf. **The Refrigerator:** The refrigerator is a white, top-freez... |
| 344795 | cake | uncertain_small_background | food | cake | ...t glow on the scene. **Central Object:** The main focus is a round, baked dish, likely a cake or a pie, resting on a metal baking sheet. The dish is completely covered with crinkled a... |
| 160012 | car | uncertain_small_background | vehicle | car | ...onment:** - The background is out of focus, but it shows a street scene at night. A white car is visible parked on the street, with its headlights or taillights glowing. There are blu... |
| 341921 | car | uncertain_small_background | vehicle | cars | ...ce, nestled among more trees. - Further in the background, beyond the trees, a few parked cars can be seen, suggesting the park is near a street or residential area. **Overall Composi... |
| 344029 | car | uncertain_small_background | vehicle | cars, hatchback, sedan | ...ther Vehicles:** - To the left, a line of parked cars is visible, including a dark blue hatchback and a silver sedan. - Further down the street, another bus, which appears to be green a... |
| 345361 | car | uncertain_small_background | vehicle | car | ...rther in the background, another house with white siding can be seen, along with a parked car and some trees or bushes. - The ground is a well-maintained green lawn. - The lighting su... |
| 369442 | car | uncertain_small_background | vehicle | van | ...rcial or institutional structure. - To the left of the building, a green and white bus or van is partially visible behind some trees. - A blue and white circular traffic sign (a "no e... |
| 433774 | car | uncertain_small_background | vehicle | cars | ...sense of perspective. The street is paved and appears empty of traffic, with a few parked cars visible far down the road. On the left side of the street, a traffic light is suspended o... |
| 424721 | carrot | uncertain_small_background | food | carrots | ...- **On the right side of the pan, extending toward the front:** A bundle of bright orange carrots, still attached to their green, leafy tops. The carrots are long and slender, and their v... |
| 423123 | cell phone | uncertain_small_background | electronics_appliance | phone | ...is reaching into the fridge, while their left hand holds a small, dark object, possibly a phone or a remote control. - Behind them, another person in a camouflage-patterned uniform (gre... |
| 7818 | chair | uncertain_small_background | furniture | chair | ... colors are white (tablecloth, flowers, napkins, plates) and black/dark gray (background, chair backs, table edges). The warm glow from beneath the vase adds a touch of golden light, cr... |
| 177861 | chair | uncertain_small_background | furniture | chairs | ...ear the center. - To the right, a small outdoor seating area with metal-framed tables and chairs is visible, partially covered by a canopy or awning. A black horse-drawn carriage is park... |
| 379476 | chair | uncertain_small_background | furniture | chairs | ...tends far into the background, where the faint glow of more lights and the silhouettes of chairs or benches can be seen, suggesting a long, continuous dining area. The perspective is enh... |
| 386352 | chair | uncertain_small_background | furniture | chair | ...eparating the court area from the spectator seating. - Behind the railing, a blue plastic chair or bench is partially visible. - The setting appears to be an outdoor or well-lit indoor ... |
| 181969 | couch | uncertain_small_background | furniture | couch | ...on a light-colored, possibly beige or cream-colored surface, which appears to be a bed or couch with a soft, textured fabric. The surface has some visible creases and folds, suggesting ... |
| 157767 | cup | uncertain_small_background | kitchen_tableware | cup | ...ir, wearing a dark blazer over a collared shirt and dark trousers. He is holding a yellow cup or glass in his right hand and appears to be speaking or gesturing with his left hand. A ... |
| 487583 | cup | uncertain_small_background | kitchen_tableware | cup | ...attice pattern. The table appears to be of a small, personal size, suitable for holding a cup or a snack. - **The Cookie:** On the table’s surface, there is a single, round, golden-br... |

## riskmask_p30_step100

### Category Decomposition

| category | removed halluc | share removed halluc | added halluc | net halluc | removed correct | share removed correct | added correct | net correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uncertain_small_background | 180 | 0.5573 | 149 | -31 | 80 | 0.6107 | 72 | -8 |
| text_logo_graphic | 68 | 0.2105 | 43 | -25 | 29 | 0.2214 | 34 | 5 |
| ordinary_visual_object | 61 | 0.1889 | 59 | -2 | 22 | 0.1679 | 34 | 12 |
| color_attribute_as_object | 14 | 0.0433 | 10 | -4 | 0 | 0.0000 | 0 | 0 |

### Flag Decomposition

| flag | removed halluc | added halluc | net halluc | removed correct | added correct | net correct |
| --- | --- | --- | --- | --- | --- | --- |
| text_or_graphic | 72 | 43 | -29 | 29 | 34 | 5 |
| color_attribute | 14 | 10 | -4 | 0 | 0 | 0 |
| uncertain_small_background | 238 | 182 | -56 | 104 | 95 | -9 |
| uppercase_mention | 4 | 2 | -2 | 1 | 2 | 1 |

### Removed Hallucinated: Text/Logo/Graphic

| image | object | category | group | terms | context |
| --- | --- | --- | --- | --- | --- |
| 84270 | airplane | text_logo_graphic | vehicle | airplane | ...bers include "219-224" and "225-232". The signs also feature small icons, including a red airplane symbol and a blue airplane symbol. To the left, a green sign with a white "E" (likely ind... |
| 161609 | airplane | text_logo_graphic | vehicle | airplane | ...n is visible. It reads “Millbrae” in large, white, sans-serif letters, with a small white airplane icon and the letters “BART” to the right of it. To the right of “Millbrae,” there is a wh... |
| 56288 | apple | text_logo_graphic | food | Apple | ...adsheet with text and lines, though the content is not legible. - The monitor has a green Apple logo sticker on its lower bezel, suggesting it is an Apple computer. - In front of the mo... |
| 432085 | apple | text_logo_graphic | food | apple | ...t. - A box of tissues with a colorful, cartoonish design (featuring what looks like a red apple and a yellow sun) is placed behind the water bottle. - A small, clear plastic cup is part... |
| 579902 | backpack | text_logo_graphic | accessory | backpack | ... sleeves and shoulders, and blue jeans. - The rider is carrying a large, brightly colored backpack or saddlebag that is strapped to the back of the motorcycle. The bag has a vibrant, geome... |
| 463802 | bear | text_logo_graphic | animal | bear | ..., dark brown, textured object that strongly resembles the head and upper torso of a teddy bear. - The material appears to be a coarse, woven fabric, possibly a type of felt or heavy kn... |
| 170595 | bed | text_logo_graphic | furniture | bed | ...l impression is one of a simple, satisfying, and comforting breakfast, perhaps enjoyed in bed on a lazy morning. There is no visible text or branding in the image. |
| 48924 | bicycle | text_logo_graphic | vehicle | bike | ... is angled slightly toward the viewer, with its front wheel and handlebars prominent. The bike is equipped for touring or off-road travel, featuring: - A black rear fender and a blac... |
| 236599 | bird | text_logo_graphic | animal | bird | ...Kite:** Dominating the upper center is a large, multi-colored kite shaped like a stylized bird or dragon with long, flowing tail feathers. It has a white body with blue and red accents... |
| 360097 | bird | text_logo_graphic | animal | bird | ...om right corner, there is a faint, stylized black signature or watermark that resembles a bird or a cursive letter “S”. - The overall lighting is even and diffuse, suggesting an overca... |
| 533493 | bird | text_logo_graphic | animal | bird | ...sses and a black t-shirt with a small, white, stylized logo on the left chest (possibly a bird or a similar figure). He is holding a white frisbee in his left hand, which is raised and... |
| 360097 | boat | text_logo_graphic | vehicle | ferry, ship | ...is a high-angle, top-down photograph of a concrete dock or pier area, likely at a port or ferry terminal. The scene is dominated by piles of luggage and a prominent white railing. **Fo... |
| 369503 | boat | text_logo_graphic | vehicle | sailboat | ... with a white horizontal stripe near the hem, which features a small red star and a white sailboat logo. Their left arm is bent, and their hand is near their neck or shoulder. Their dark p... |
| 54654 | book | text_logo_graphic | indoor_other | book | ... the words "my raggedy" and "don't" can be discerned, suggesting it might be a children's book or poster. To the right of the person, a white door or cabinet is visible, with several ... |
| 119365 | book | text_logo_graphic | indoor_other | book | ...e center, a poster with a blue border and a speech bubble graphic shows a child reading a book. - To the right, a prominent white poster with a blue border is titled "Building vocabula... |
| 251824 | bottle | text_logo_graphic | kitchen_tableware | Bottle, bottle, bottles | ...age captures a moment of focused, hands-on crafting, likely involving repurposing plastic bottles into a new object, possibly a toy, a tool, or a piece of art. |
| 22892 | bowl | text_logo_graphic | kitchen_tableware | container | ...e a young tomato or pepper plant, with several stems and leaves. It is planted in a small container filled with dark soil. The pot is partially visible, showing a yellow and white label wit... |
| 194471 | bowl | text_logo_graphic | kitchen_tableware | bowl | ...ment of athletic motion. He is riding a skateboard on the curved edge of a concrete skate bowl. - He is wearing a black short-sleeved t-shirt with a white graphic on the chest, light-w... |
| 309678 | bowl | text_logo_graphic | kitchen_tableware | container | ...ems, possibly carrots or sweet potatoes. - In the upper right background, a clear plastic container with a label is partially visible. The word “CINNAMON” can be seen on the label, suggesti... |
| 28993 | car | text_logo_graphic | vehicle | cars | ...ue to the depth of field. To the left, a street runs parallel to the sidewalk, with a few cars and street signs visible. A blue and white vertical sign is mounted on a pole on the left... |
| 109900 | car | text_logo_graphic | vehicle | SUV | ... in discussion or waiting. - **Vehicle:** To the right, a white, rugged, four-wheel-drive SUV is parked. It has a black roof rack and a tall antenna on the front. The vehicle is clear... |
| 143572 | car | text_logo_graphic | vehicle | car | ...printed in large, white, sans-serif letters. - To the left, the logo and text “esurance car insurance” are visible. - To the right, the URL “usopen.org” is displayed in white. - A... |
| 336628 | car | text_logo_graphic | vehicle | Car, car | ...an in a highly stylized and colorful costume standing on the open-air platform of a cable car, likely during a parade or festive street event. **The Central Figure:** The man is the ... |
| 460929 | car | text_logo_graphic | vehicle | car | This is a candid, sunlit photograph of a casual meal, likely taken from inside a car, as suggested by the window in the background. The scene is composed of a hot dog and a b... |
| 468965 | car | text_logo_graphic | vehicle | car | ... white design with a large red letter “E” and cartoon-like illustrations, including a red car and a blue bird. - **Child in the Foreground (Left Side):** In the lower-left foreground... |

### Removed Hallucinated: Uncertain/Small/Background

| image | object | category | group | terms | context |
| --- | --- | --- | --- | --- | --- |
| 110884 | airplane | uncertain_small_background | vehicle | airplane | ...s is a photograph of a compact, utilitarian bathroom, likely from a hotel room, train, or airplane cabin, characterized by its small size and functional design. The image has a strong gree... |
| 56344 | apple | uncertain_small_background | food | Apple | ... is a slim, white, full-sized keyboard with a standard QWERTY layout. It appears to be an Apple Magic Keyboard. - To the right of the keyboard is a white, ergonomic computer mouse, like... |
| 147205 | apple | uncertain_small_background | food | apple | ... submerged in the sauce. - A single, thin slice of what appears to be caramelized pear or apple is placed beside the chestnuts, adding a contrasting color and texture. **The Plate and ... |
| 210388 | backpack | uncertain_small_background | accessory | backpack | ...N” and possibly “SUMMIT” below it. He is also wearing yellow gloves and has a small, dark backpack or bag slung over his shoulder. * **On the right:** A woman with shoulder-length brown ... |
| 57150 | bear | uncertain_small_background | animal | bear | ...laid skirt, white tights, and dark shoes. She is holding a very large, fluffy white teddy bear that towers over her. The bear is oversized, with a prominent red bow tied around its nec... |
| 231879 | bear | uncertain_small_background | animal | bear | ...entle, focused expression. The cake is decorated with a small, brown figurine (possibly a bear or animal) on top and a single, thin candle. **Background Elements:** - Behind the main ... |
| 315492 | bear | uncertain_small_background | animal | Bear, bear | ...nded toward it, suggesting she is either pulling a sheet or adjusting the roll. - **Teddy Bear:** On the floor, to the child’s left, sits a large, brown teddy bear. The bear is lying o... |
| 370478 | bear | uncertain_small_background | animal | bears | ...rter or sheet featuring a repeating pattern of small, dark cartoon-like figures (possibly bears or animals) and stars. - A black jacket or piece of clothing is draped over the edge of t... |
| 368038 | bed | uncertain_small_background | furniture | bed | ...green tree. **Environment and Background** * **Ground:** The cabooses sit on a gravel bed next to a patch of grassy, slightly uneven ground. The grass is a mix of green and brown,... |
| 575081 | bed | uncertain_small_background | furniture | bed | ...re strewn across the floor, connecting the gaming equipment. A small, woven basket or pet bed is on the floor near the boy’s feet. A small, dark-colored mug sits on a shelf on the ent... |
| 87038 | bench | uncertain_small_background | outdoor_sign | bench | ...nd is riding a small, black-and-white BMX bike. The bike is positioned near a low, wooden bench or platform. - The ground is a flat, gray asphalt surface, typical of a skatepark. Severa... |
| 145020 | bench | uncertain_small_background | outdoor_sign | benches | ...scene. On the far left, a group of visitors, including children and adults, are seated on benches or standing near the signpost. In the middle ground, a few people are walking or standing... |
| 319369 | bench | uncertain_small_background | outdoor_sign | bench | ...the third in a white top and dark shorts. - To the left, a man is sitting on a low wooden bench or stool, wearing sunglasses and a patterned shirt. He appears to be relaxing. - To the r... |
| 379476 | bench | uncertain_small_background | outdoor_sign | benches | ...into the background, where the faint glow of more lights and the silhouettes of chairs or benches can be seen, suggesting a long, continuous dining area. The perspective is enhanced by th... |
| 411530 | bench | uncertain_small_background | outdoor_sign | bench | ...k, and gray—and are waiting to board the lift. Some are standing, others are sitting on a bench near a small building. **Midground:** - A chairlift system is the central focus. It cons... |
| 431896 | bench | uncertain_small_background | outdoor_sign | benches | ... warm glow. Further down the platform, other station infrastructure is visible, including benches and what appears to be a ticket machine or information kiosk. - **Spatial Relationship:**... |
| 500270 | bench | uncertain_small_background | outdoor_sign | bench | ...isible in the background of the skatepark, also in silhouette. One person is sitting on a bench or ledge, while others are scattered around the various features of the park. **Backgrou... |
| 476491 | bicycle | uncertain_small_background | vehicle | bike | ...oreground is a dark, textured asphalt or concrete surface, likely a pedestrian walkway or bike lane. It is marked with large, bold yellow Japanese characters: “スクールゾーン” (Sukūru Zōn), w... |
| 534827 | bicycle | uncertain_small_background | vehicle | bike, bikes | ... strong backlighting creates silhouettes and highlights the edges of the riders and their bikes. - Long, dark shadows are cast by the motorcycles and riders onto the road, stretching to... |
| 344100 | bird | uncertain_small_background | animal | turkey | ...made with whole wheat or multigrain bread, and the filling appears to be layers of ham or turkey and a slice of white cheese, with some of the filling slightly oozing out. - To the left ... |
| 400922 | bird | uncertain_small_background | animal | bird | ...is a vibrant blue, filled with large, fluffy white clouds. A small, dark object, likely a bird or a drone, is captured in mid-flight against the sky, positioned above the KeyBank sign.... |
| 296284 | boat | uncertain_small_background | vehicle | liner | ...he ambient light. The background behind the donuts on each shelf is a patterned paper or liner with a repeating orange and white geometric design, possibly a stylized checkered or diam... |
| 304812 | boat | uncertain_small_background | vehicle | boats, ships | ...ark shapes against the bright sky. On the far left horizon, a few small, distant ships or boats can be seen on the water, adding a sense of scale and distance to the scene. **Overall I... |
| 463527 | boat | uncertain_small_background | vehicle | liner | ...irplane meal tray, likely from a premium or business class cabin, set on a dark blue tray liner with a subtle striped pattern. The tray is positioned on an airplane seat, with the windo... |
| 540932 | boat | uncertain_small_background | vehicle | sailboard | ...submerged or resting on the sand. It appears to be a piece of equipment, possibly a small sailboard or a kiteboard, with a black and yellow board attached to it. A person is sitting on the ... |

### Removed Correct: Text/Logo/Graphic

| image | object | category | group | terms | context |
| --- | --- | --- | --- | --- | --- |
| 185599 | apple | text_logo_graphic | food | apple, apples | ...red. - **Center (Background):** Nestled between the two foreground fruits is a cluster of apples. One apple is visible directly behind the left orange, showing a blend of red and yellow ... |
| 219283 | banana | text_logo_graphic | food | BANANAS, Bananas, banana, bananas | ... has a large, rectangular cutout on its side and a smaller circular hole. The text “ALITY BANANAS” is visible on its side, suggesting it was used for shipping bananas. Below this, the wor... |
| 97988 | bench | text_logo_graphic | outdoor_sign | Bench, bench | ...ly more stylized font. The banner is slightly wrinkled, indicating it is fabric. - **Park Bench:** To the right of the man, a green park bench is visible. Two people are seated on it, p... |
| 206994 | bench | text_logo_graphic | outdoor_sign | bench | ...-roofed shelter with a metal frame stands on the platform. Inside the shelter, there is a bench with blue and yellow seating. A small, yellow sign with a blue border is mounted on the s... |
| 481390 | bench | text_logo_graphic | outdoor_sign | bench | ... on the sideline, including a man in a black jacket and another in a white shirt near the bench area. **The Players and Their Actions:** There are eight players clearly visible on the ... |
| 459500 | bird | text_logo_graphic | animal | Birds, bird, birds, herons | ... dark, rectangular pedestal. The sculpture depicts three stylized birds, likely cranes or herons, rendered in a metallic, weathered material that appears to be steel or iron. Their forms... |
| 396274 | broccoli | text_logo_graphic | food | broccoli | ...robust plants with broad, dark green, slightly crinkled leaves. These are identifiable as broccoli plants, with some showing developing green florets. The leaves have prominent, lighter-co... |
| 86483 | car | text_logo_graphic | vehicle | car | ...e of the sign on the left. Further in the background, more trees and a glimpse of a white car parked on a street can be seen. - **Foreground:** The sign sits on a low, stone-faced wal... |
| 40083 | chair | text_logo_graphic | furniture | chair, stool | ...se of everyday urban life. **Main Subjects:** - **Man on the left:** Seated in a folding chair, he is wearing a dark, long-sleeved shirt and dark pants. He has his legs crossed and is ... |
| 481390 | chair | text_logo_graphic | furniture | seats | ...sible on the red sideline barrier. * The “Coca-Cola” logo is visible on the black bench seats in the background. **Overall Composition:** The photograph captures a moment of intense ... |
| 189752 | cup | text_logo_graphic | kitchen_tableware | cup | ... long-sleeved shirt and appear to be eating or drinking from a straw. A small white paper cup with red and black text is on the table in front of them. The text on the cup is partiall... |
| 555009 | cup | text_logo_graphic | kitchen_tableware | Cup, cup | ... the keyboard. The headphones are not on anyone's head. * **Other Objects:** * **Cup:** A green and white patterned ceramic mug or cup sits on the desk between the two monito... |
| 157601 | dining table | text_logo_graphic | furniture | Table, table | ...fé or coffee shop. The image is taken from a low angle, looking up at the subject and the table, which creates a sense of intimacy and immediacy. **Main Subject:** - A person with shor... |
| 160012 | dining table | text_logo_graphic | furniture | table | ... board, with another pizza partially visible in the background. A person is seated at the table, interacting with the food. **Foreground and Center:** - The main subject is a large, ro... |
| 231097 | dining table | text_logo_graphic | furniture | dining table, table | This is a low-light, close-up photograph of a partially eaten meal on a dining table, conveying a sense of intimacy and casual dining. **Main Subject: The Plate of Food** - ... |
| 277689 | dining table | text_logo_graphic | furniture | Table, table | This is a vibrant, sun-drenched photograph of an outdoor wedding or celebration cake table, set against a tropical backdrop. The composition is carefully arranged to draw the eye t... |
| 261888 | horse | text_logo_graphic | animal | horse, horses | ...a wide-angle, eye-level photograph capturing a cyclist paused on a paved road, facing two horses that are walking toward him in the distance. The scene is set on a bright, sunny day unde... |
| 93437 | person | text_logo_graphic | person | Man, man | This is a candid, indoor photograph of an elderly man, likely in his 70s or 80s, who is the central focus of the image. He is smiling broadly a... |
| 97337 | person | text_logo_graphic | person | people | ...eddish-brown wooden stand. The TV screen is on, displaying a black-and-white image of two people in what appears to be a dynamic, possibly athletic or dance-like pose. The image is somew... |
| 133000 | person | text_logo_graphic | person | driver | ...ame. - The front carriage is clearly visible, featuring a large, curved windshield with a driver inside. The train’s number, “31,” is displayed in white on a black background above the w... |
| 182021 | person | text_logo_graphic | person | player, players | ...field: the green of the grass, the brown of the dirt, and the dark and light tones of the players' uniforms. * The overall impression is one of action and anticipation, capturing a flee... |
| 330818 | person | text_logo_graphic | person | man, person | This is a candid, slightly grainy photograph of a person working in a commercial kitchen, likely a diner or small restaurant, captured from a slig... |
| 369812 | person | text_logo_graphic | person | people | ...ble-decker bus is visible, a classic symbol of London. - Other cars and a bus stop with people waiting can be seen in the distance. - **Sidewalk and Pedestrians:** A paved sidewalk ru... |
| 393056 | person | text_logo_graphic | person | person, rider | ...y. **Main Subject:** The central focus is a single surfer riding a wave. The surfer is a person with short, light-colored hair, likely blond or light brown, visible above the collar of ... |
| 462643 | person | text_logo_graphic | person | person | This is a close-up, angled photograph of a person’s hand holding a disassembled or partially disassembled smartwatch, revealing its interna... |

### Removed Correct: Uncertain/Small/Background

| image | object | category | group | terms | context |
| --- | --- | --- | --- | --- | --- |
| 161609 | backpack | uncertain_small_background | accessory | backpack | ...lack or charcoal grey, hooded sweatshirt or vest over a white t-shirt. A large, dark grey backpack is slung over their shoulders. The most striking feature is a small, tan or light brown d... |
| 341921 | backpack | uncertain_small_background | accessory | backpack | ...ath. - To the right of the boy, on the ground next to the bench, lies a black, soft-sided backpack or small bag. **Midground:** - The area directly behind the boy and extending to the lef... |
| 425221 | backpack | uncertain_small_background | accessory | backpack | ...n is seen in silhouette, standing and looking out the window. They appear to be wearing a backpack and are turned slightly away from the camera, gazing toward the right. - To the right, an... |
| 370375 | baseball bat | uncertain_small_background | sports | baseball bat | ...small, stylized logo or design on the front. - The child is holding a bright blue plastic baseball bat with a silver-colored barrel. The grip of the bat is black with a textured, possibly spir... |
| 134722 | bench | uncertain_small_background | outdoor_sign | bench | ...f with a red trim. A red downpipe runs vertically down the side of the brick structure. A bench is visible on the platform near this building. **Background and Environment:** Behind th... |
| 336209 | bench | uncertain_small_background | outdoor_sign | Bench, bench | ...bove it. The ramp is constructed from plywood and has a rough, unfinished look. - **Metal Bench:** To the right of the ramp, a long, weathered metal bench with a wooden top is visible. ... |
| 419408 | bench | uncertain_small_background | outdoor_sign | bench | ...ject (Observer):** - To the right, another young man is seated on a dark green metal park bench. He is casually dressed in a blue jacket over a red shirt and dark pants. - He is looking... |
| 459809 | boat | uncertain_small_background | vehicle | sailboats, vessels | ...ndmass or mountain range forms the horizon line. A few small, indistinct shapes, possibly sailboats or other vessels, are visible on the water near the left edge of the frame. **Foreground... |
| 396526 | book | uncertain_small_background | indoor_other | book | ... A small, dark wooden side table sits beside the lamp. On it rests a closed, dark-colored book or notebook. * **Door and View:** A large, white-framed glass door or window is set int... |
| 458410 | book | uncertain_small_background | indoor_other | books | ...h a white tablecloth and has various items on it, including what appears to be a stack of books or magazines, a green box, and other small objects. The table is supported by metal legs.... |
| 226984 | bottle | uncertain_small_background | kitchen_tableware | bottle, bottles | ...the sink. - **Above the Stove:** A decorative wooden shelf or rack holds various jars and bottles, possibly spices or condiments. - **Wall Decor:** A framed picture hangs on the wall abov... |
| 91779 | bowl | uncertain_small_background | kitchen_tableware | container, containers | ... cup, possibly for water or soda. - Next to the cup, there are two small, white plastic containers with lids. One contains a yellowish substance (possibly mustard or a dipping sauce), and ... |
| 150417 | bowl | uncertain_small_background | kitchen_tableware | bowl, container | ...s:** In the foreground, a large blue plastic object (possibly a vacuum cleaner or a large container) is partially visible on the left. A white plastic bag is also on the counter near the bl... |
| 157418 | bowl | uncertain_small_background | kitchen_tableware | bowl, containers | ...rom the side or front, casting soft shadows and highlighting the textures of the food and containers. - The table surface is visible at the bottom of the frame, covered with a yellow and whi... |
| 447522 | broccoli | uncertain_small_background | food | Broccoli, broccoli | ... is a colorful mixture of sautéed or stir-fried vegetables. The most prominent are: - **Broccoli**: Several florets of green broccoli are scattered throughout the dish. Some are whole sm... |
| 344795 | cake | uncertain_small_background | food | cake | ...t glow on the scene. **Central Object:** The main focus is a round, baked dish, likely a cake or a pie, resting on a metal baking sheet. The dish is completely covered with crinkled a... |
| 228436 | car | uncertain_small_background | vehicle | cars | ... metal, with vertical posts and horizontal bars. - On the left side of the canal, several cars are parked along the street, partially obscured by the trees. - The trees are bare, with ... |
| 260470 | car | uncertain_small_background | vehicle | car | ...shirt is visible on the left, and another person in a grey shirt is further back. A white car is partially visible through the glass, suggesting the bakery is located on a street. The... |
| 345361 | car | uncertain_small_background | vehicle | car | ...rther in the background, another house with white siding can be seen, along with a parked car and some trees or bushes. - The ground is a well-maintained green lawn. - The lighting su... |
| 369442 | car | uncertain_small_background | vehicle | van | ...rcial or institutional structure. - To the left of the building, a green and white bus or van is partially visible behind some trees. - A blue and white circular traffic sign (a "no e... |
| 7818 | chair | uncertain_small_background | furniture | chair | ... colors are white (tablecloth, flowers, napkins, plates) and black/dark gray (background, chair backs, table edges). The warm glow from beneath the vase adds a touch of golden light, cr... |
| 115245 | chair | uncertain_small_background | furniture | chairs | ...is characterized by light-colored wooden flooring and long, white tables with red plastic chairs, suggesting a communal or functional space. **Foreground and Center:** - A large, black,... |
| 139684 | chair | uncertain_small_background | furniture | seat | ...wn or black dog is curled up and sleeping on the right side of the sofa, partially on the seat and partially on the backrest. A light-colored, possibly beige or cream, throw blanket is... |
| 177861 | chair | uncertain_small_background | furniture | chairs | ...ear the center. - To the right, a small outdoor seating area with metal-framed tables and chairs is visible, partially covered by a canopy or awning. A black horse-drawn carriage is park... |
| 259597 | chair | uncertain_small_background | furniture | chairs | ... boy as the main subject. The people are dressed in casual attire, and some are seated on chairs visible to the left. **Background:** - The background is softly blurred but reveals the ... |
