# RiskMask Selection Enrichment Probe

## Inputs

- trace_jsonl: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/probes/results/riskmask_selection_enrichment/8b_tr075_p30/pair_kl_trace.jsonl`
- rank_scope: `global`
- top_p: 0.3000
- records: 990
- tokens: 582153
- labeled object mentions: 16435
- base hallucination rate: 0.1614

## Token Selection

| mode | valid | selected | selected frac | threshold |
| --- | --- | --- | --- | --- |
| riskmask_nll | 582153 | 174646 | 0.3000 | 0.4658 |
| riskmask_entropy | 582153 | 174646 | 0.3000 | 0.4688 |
| rkl_only | 582153 | 174646 | 0.3000 | 0.7000 |
| nll_only | 582153 | 174646 | 0.3000 | 0.7000 |
| entropy_only | 582153 | 174646 | 0.3000 | 0.7000 |
| random | 582153 | 174646 | 0.3000 | 0.7000 |

## Object Mention Enrichment

| mode | span selected | span precision | span recall | span correct sel. | span lift | token selected | token precision | token recall | token lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| riskmask_nll | 4931 | 0.2154 | 0.4003 | 0.2807 | 1.3342 | 4699 | 0.2066 | 0.3660 | 1.2801 |
| riskmask_entropy | 4931 | 0.2192 | 0.4075 | 0.2793 | 1.3581 | 4650 | 0.2118 | 0.3713 | 1.3122 |
| rkl_only | 4931 | 0.2164 | 0.4022 | 0.2804 | 1.3405 | 5468 | 0.1988 | 0.4097 | 1.2315 |
| nll_only | 4931 | 0.2206 | 0.4101 | 0.2788 | 1.3669 | 4046 | 0.2098 | 0.3200 | 1.2999 |
| entropy_only | 4931 | 0.2255 | 0.4191 | 0.2771 | 1.3970 | 3950 | 0.2261 | 0.3366 | 1.4005 |
| random | 4931 | 0.1659 | 0.3083 | 0.2984 | 1.0277 | 5532 | 0.1509 | 0.3147 | 0.9351 |

## Top RiskMask Hallucinated Mentions

| image | object | score | token frac | context |
| --- | --- | --- | --- | --- |
| 150649 | cow | 0.9988 | 1.0000 | ...tire inner surface. Prominent features include: - A large, stylized, cartoonish bull’s head with horns, rendered in shades of yellow, green, and white, located near... |
| 124636 | person | 0.9981 | 1.0000 | ...ng distinct shadows on the ground.  **Main Subjects:**  1.  **The Standing Cow (Mother):**     *   **Appearance:** This is a large, robust cow with a predominantly da... |
| 126107 | dog | 0.9938 | 1.0000 | ... the water’s edge, while another is seated or crouching nearby, possibly with a dog. A third figure is further to the right, near the boat. - The ocean meets the s... |
| 105014 | orange | 0.9921 | 1.0000 | ...lored beans or grains, possibly lentils or chickpeas. Mixed in are small, diced orange carrots and finely chopped green scallions or chives. - **Vegetables and Greens... |
| 564133 | person | 0.9913 | 1.0000 | ...ht foreground, almost directly facing the camera. It is a large adult, likely a male given its size and the presence of two long, curved tusks. Its trunk is slightl... |
| 411530 | orange | 0.9907 | 1.0000 | ...i lift station. They are dressed in colorful winter gear—jackets in green, red, orange, black, and white—and are either waiting to board the lift or have just disemba... |
| 211674 | orange | 0.9876 | 1.0000 | ...the front bumper. The sign displays the text “RUTA DO ARROZ” in large, stylized orange and red letters. To the left of the text is a graphic of a golden-yellow rice s... |
| 40083 | cell phone | 0.9875 | 1.0000 | ...this, smaller text is partially visible, possibly indicating a business name or phone number. - **Background:**   - Behind the men, a dark-colored sedan is parked on... |
| 360951 | clock | 0.9865 | 1.0000 | ...l, there are a few small decorative items, including a small, framed picture or clock and a small, round, metallic object. - **Wall:** The wall behind the desk is pa... |
| 514376 | train | 0.9859 | 1.0000 | ...capturing a modern street scene in an urban environment, featuring a light rail train as the central subject. The image is composed with the train moving from the ri... |
| 40083 | dining table | 0.9840 | 1.0000 | ...isible sign with a red and white design.   - On the far right, there is a small table with a sign that reads “电子” (Electronic), suggesting it might be a small electr... |
| 177893 | person | 0.9832 | 1.0000 | ...een the words “FINAL” and “DESTINATION,” there is a stylized white graphic of a person with arms outstretched, resembling a figure in motion or a person falling.   - ... |
| 562229 | bench | 0.9832 | 1.0000 | ... boy.   - **Playground Equipment:** To the right, in the background, there is a bench or a piece of playground equipment, and two people (possibly children) are sitt... |
| 468965 | airplane | 0.9829 | 1.0000 | ...rounded by blue, green, and yellow patterns, and what appears to be a small red airplane or rocket. - The boy on the right holds a simpler, green and white kite, possib... |
| 35326 | orange | 0.9824 | 1.0000 | ...i-tiered spice rack. It holds numerous spice jars with colorful, mostly red and orange, labels. The rack has a vertical column of blue-capped jars on the left and a t... |
| 432085 | carrot | 0.9810 | 1.0000 | ... what looks like shredded orange-colored meat or vegetable (possibly chicken or carrots) and a creamy white sauce. The bread is a light, golden-brown color. The plate ... |
| 257896 | cell phone | 0.9801 | 1.0000 | .... The image has a warm, slightly grainy quality, suggesting it was taken with a smartphone or compact camera in low light.  **The Subject:** The central figure is a man w... |
| 226171 | car | 0.9800 | 1.0000 | ...r, with a large featured image at the top showing a nighttime street scene with cars and streetlights. Below this, there are several smaller thumbnail images arrang... |
| 200839 | car | 0.9786 | 1.0000 | ..., there are decorative stripes in pink, green, and blue. The bus has a standard cab at the front and a long, boxy body. Its wheels are visible, and it appears to b... |
| 177861 | horse | 0.9782 | 1.0000 | ...restaurant area is visible on the right, with a metal table and chairs. A black horse-drawn carriage is parked behind the seating area, with its wheels and harness v... |
| 56288 | dining table | 0.9779 | 1.0000 | ...lack electronic device, possibly a phone or remote control, is visible.   - The desk surface is dark brown wood, and the plate is sitting on a slightly reflective, ... |
| 281414 | chair | 0.9747 | 1.0000 | ...focus elements of an urban or campus environment. One can discern the shapes of chairs (possibly patio furniture) and the side of a white vehicle, perhaps a van or tr... |
| 245173 | bench | 0.9739 | 1.0000 | ...ainted facade is visible. It has a dark roof and a small, rectangular window. A bench is placed in front of this building. - **Right Side:** To the right, there are ... |
| 171757 | cell phone | 0.9735 | 1.0000 | ... and gray) and dark pants. They are holding something in their hands, perhaps a phone or a small object, and seem to be looking down at it or toward the person on th... |
| 497344 | dining table | 0.9726 | 1.0000 | ...graph of a young toddler, likely between 1 and 2 years old, sitting at a wooden table and interacting with technology in a manner that is both adorable and amusingly... |
| 555009 | bowl | 0.9719 | 1.0000 | ...wo main monitors, with a small orange knob or button on top.     *   **Drinking Container:** A green and white patterned ceramic mug or tumbler is on the right side of t... |
| 463527 | orange | 0.9716 | 1.0000 | ...ntains a fresh salad, primarily composed of shredded green lettuce and shredded orange carrots, with some other green leafy vegetables mixed in. The salad appears to ... |
| 173091 | bowl | 0.9711 | 1.0000 | ...of the label, possibly indicating a price or a promotional code.  2.  **Plastic Container with Sandwich:**     *   Located to the left of the frappuccino bottle.     *  ... |
| 150638 | bowl | 0.9700 | 1.0000 | ...ntains a pale, cloudy liquid, possibly a cocktail or smoothie. A yellow plastic container, likely holding fuel or another liquid, is attached to the side of the engine. ... |
| 334483 | dining table | 0.9692 | 1.0000 | ... hair, wearing sunglasses perched on her head, is standing and leaning over the table. She is wearing a blue and white striped tank top with a yellow accent and blac... |
