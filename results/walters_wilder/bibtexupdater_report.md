# bibtexupdater on the Walters & Wilder ChatGPT-citation supplement

- Entries: **341** (supported type = journal articles); coverage 99.7%
- Detection rate: **0.964**  
- False-positive rate: **0.145**  
- F1 (hallucination): **0.913**  
- Tier-weighted F1: **0.960**  
- ECE: 0.20562056470588233

## Per tier

| Tier | n_hall | n_valid | detection_rate | FPR | F1 |
|---|---|---|---|---|---|
| 2 | 17 | 172 | 0.824 | 0.145 | 0.500 |
| 3 | 152 | 172 | 0.980 | 0.145 | 0.914 |

## Per hallucination type

| Type | count | detection_rate |
|---|---|---|
| valid | 172 | 0.000 |
| plausible_fabrication | 139 | 0.993 |
| near_miss_title | 13 | 0.846 |
| swapped_authors | 12 | 0.750 |
| wrong_venue | 5 | 1.000 |

## By GPT version

| Version | n | detection_rate | FPR |
|---|---|---|---|
| GPT-3.5 | 119 | 0.990 | 0.053 |
| GPT-4 | 222 | 0.928 | 0.157 |

## By subject field

| Field | n | detection_rate | FPR |
|---|---|---|---|
| Humanities | 24 | 1.000 | 0.000 |
| Natural sciences | 103 | 0.935 | 0.211 |
| Social sciences | 214 | 0.972 | 0.120 |

## Tool status distribution

| Status | count |
|---|---|
| partial_match | 107 |
| verified | 104 |
| unconfirmed | 48 |
| not_found | 34 |
| author_truncated | 18 |
| title_mismatch | 13 |
| author_mismatch | 8 |
| venue_mismatch | 4 |
| year_mismatch | 3 |
| hallucinated | 1 |
| MISSING | 1 |

## False positives on real articles (25)

- `ww_35_t23_c2301` [venue_mismatch] Managing cultural diversity: Implications for organizational competitiveness
- `ww_4_t9_c909` [title_mismatch] Retirement migration counties in the southeastern United States: Geographic, dem
- `ww_4_t10_c1010` [author_truncated] Estimating the effect of the one-child policy on the sex ratio imbalance in Chin
- `ww_4_t10_c1014` [author_mismatch] Options for fertility policy transition in China
- `ww_4_t17_c1706` [partial_match] Will shift to remote teaching be boon or bane for online learning?
- `ww_4_t17_c1708` [author_truncated] Student perception of helpfulness of facilitation strategies that enhance instru
- `ww_4_t19_c1902` [author_truncated] The impact of the National School Lunch Program on child health: A nonparametric
- `ww_4_t21_c2104` [author_truncated] On the validity of student evaluation of teaching: The state of the art
- `ww_4_t23_c2301` [author_truncated] Context and leadership: an examination of the nine-factor full-range leadership 
- `ww_4_t23_c2309` [title_mismatch] Leadership skills for a changing world: Solving complex social problems
- `ww_4_t29_c2906` [author_truncated] Factors associated with home advantage in English and Scottish soccer matches
- `ww_4_t29_c2908` [title_mismatch] Does the home advantage depend on the crowd support? Evidence from same-stadium 
- `ww_4_t32_c3203` [author_truncated] Psychological testing and the selection of police officers: A national survey
- `ww_4_t34_c3402` [title_mismatch] The potential role of dentists in identifying patients' risk of experiencing cor
- `ww_4_t34_c3403` [year_mismatch] Periodontal disease and atherosclerotic vascular disease: Does the evidence supp
- `ww_4_t35_c3501` [author_truncated] The effect of tobacco ingredients on smoke chemistry, Part I: flavourings and ad
- `ww_4_t35_c3505` [author_truncated] Levels of selected carcinogens and toxicants in vapour from electronic cigarette
- `ww_4_t36_c3603` [title_mismatch] Insomnia in primary care: Misreported, mishandled, and just plain missed
- `ww_4_t36_c3605` [not_found] Medical and socio-professional impact of insomnia
- `ww_4_t37_c3710` [author_truncated] Reintroducing resurrected species: Selecting DeExtinction candidates
- `ww_4_t37_c3711` [title_mismatch] Mammoth 2.0: will genome editing resurrect extinct species?
- `ww_4_t38_c3804` [author_truncated] Four decades of forest persistence, clearance and logging on Borneo
- `ww_4_t39_c3901` [partial_match] Human alteration of the global nitrogen and phosphorus soil balances for the per
- `ww_4_t40_c4007` [author_mismatch] Comparative environmental life cycle assessment of conventional and electric veh
- `ww_4_t41_c4107` [author_truncated] Adaptation to flood risk: Results of international paired flood event studies

## Missed hallucinations (6)

- `ww_35_t28_c2802` [swapped_authors, tool=verified] Low-level environmental lead exposure and children's intellectual function: an i
- `ww_4_t9_c910` [swapped_authors, tool=unconfirmed] Going home” or “leaving home”? The impact of person and place ties on anticipate
- `ww_4_t22_c2201` [near_miss_title, tool=unconfirmed] The characteristics of gap year students and their tertiary academic outcomes
- `ww_4_t35_c3508` [swapped_authors, tool=unconfirmed] A randomized trial of e-cigarettes versus nicotine-replacement therapy
- `ww_4_t38_c3808` [near_miss_title, tool=unconfirmed] The population and distribution of orangutans (Pongo spp.) in and around the Dan
- `ww_4_t42_c4204` [plausible_fabrication, tool=MISSING] The Development of Thorium Molten Salt Reactor (TMSR) Technology in China
