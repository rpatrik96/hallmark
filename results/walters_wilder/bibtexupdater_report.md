# bibtexupdater on the Walters & Wilder ChatGPT-citation supplement

- Entries: **341** (supported type = journal articles); coverage 100.0%
- Detection rate: **0.929**  
- False-positive rate: **0.076**  
- F1 (hallucination): **0.926**  
- Tier-weighted F1: **0.959**  
- ECE: 0.16926731964809394

## Per tier

| Tier | n_hall | n_valid | detection_rate | FPR | F1 |
|---|---|---|---|---|---|
| 2 | 17 | 172 | 0.471 | 0.076 | 0.421 |
| 3 | 152 | 172 | 0.980 | 0.076 | 0.949 |

## Per hallucination type

| Type | count | detection_rate |
|---|---|---|
| valid | 172 | 0.000 |
| plausible_fabrication | 139 | 1.000 |
| near_miss_title | 13 | 0.769 |
| swapped_authors | 12 | 0.250 |
| wrong_venue | 5 | 1.000 |

## By GPT version

| Version | n | detection_rate | FPR |
|---|---|---|---|
| GPT-3.5 | 119 | 0.990 | 0.053 |
| GPT-4 | 222 | 0.841 | 0.078 |

## By subject field

| Field | n | detection_rate | FPR |
|---|---|---|---|
| Humanities | 24 | 0.941 | 0.000 |
| Natural sciences | 103 | 0.848 | 0.123 |
| Social sciences | 214 | 0.962 | 0.056 |

## Tool status distribution

| Status | count |
|---|---|
| verified | 148 |
| partial_match | 99 |
| not_found | 36 |
| unconfirmed | 23 |
| title_mismatch | 15 |
| author_mismatch | 8 |
| year_mismatch | 6 |
| venue_mismatch | 4 |
| hallucinated | 2 |

## False positives on real articles (13)

- `ww_35_t23_c2301` [venue_mismatch] Managing cultural diversity: Implications for organizational competitiveness
- `ww_4_t9_c909` [title_mismatch] Retirement migration counties in the southeastern United States: Geographic, dem
- `ww_4_t17_c1706` [partial_match] Will shift to remote teaching be boon or bane for online learning?
- `ww_4_t18_c1802` [year_mismatch] Explaining charter school effectiveness
- `ww_4_t23_c2309` [title_mismatch] Leadership skills for a changing world: Solving complex social problems
- `ww_4_t29_c2908` [title_mismatch] Does the home advantage depend on the crowd support? Evidence from same-stadium 
- `ww_4_t34_c3402` [title_mismatch] The potential role of dentists in identifying patients' risk of experiencing cor
- `ww_4_t34_c3403` [year_mismatch] Periodontal disease and atherosclerotic vascular disease: Does the evidence supp
- `ww_4_t36_c3603` [title_mismatch] Insomnia in primary care: Misreported, mishandled, and just plain missed
- `ww_4_t36_c3605` [year_mismatch] Medical and socio-professional impact of insomnia
- `ww_4_t37_c3711` [title_mismatch] Mammoth 2.0: will genome editing resurrect extinct species?
- `ww_4_t39_c3901` [year_mismatch] Human alteration of the global nitrogen and phosphorus soil balances for the per
- `ww_4_t40_c4007` [author_mismatch] Comparative environmental life cycle assessment of conventional and electric veh

## Missed hallucinations (12)

- `ww_35_t28_c2802` [swapped_authors, tool=verified] Low-level environmental lead exposure and children's intellectual function: an i
- `ww_4_t1_c106` [swapped_authors, tool=verified] Who was buried at Stonehenge?
- `ww_4_t9_c910` [swapped_authors, tool=unconfirmed] Going home” or “leaving home”? The impact of person and place ties on anticipate
- `ww_4_t13_c1301` [swapped_authors, tool=verified] Mineral supply for sustainable development requires resource governance
- `ww_4_t22_c2201` [near_miss_title, tool=unconfirmed] The characteristics of gap year students and their tertiary academic outcomes
- `ww_4_t35_c3508` [swapped_authors, tool=unconfirmed] A randomized trial of e-cigarettes versus nicotine-replacement therapy
- `ww_4_t37_c3701` [swapped_authors, tool=unconfirmed] Has the Earth's sixth mass extinction already arrived?
- `ww_4_t38_c3808` [near_miss_title, tool=unconfirmed] The population and distribution of orangutans (Pongo spp.) in and around the Dan
- `ww_4_t38_c3812` [near_miss_title, tool=unconfirmed] Distribution and conservation status of the orangutan (Pongo spp.) on Borneo and
- `ww_4_t38_c3813` [swapped_authors, tool=unconfirmed] Land-cover changes predict steep declines for the Sumatran orangutan (Pongo abel
- `ww_4_t39_c3908` [swapped_authors, tool=unconfirmed] Special topics—Mitigation of methane and nitrous oxide emissions from animal ope
- `ww_4_t40_c4002` [swapped_authors, tool=unconfirmed] Life cycle analysis of lithium-ion batteries for automotive applications
