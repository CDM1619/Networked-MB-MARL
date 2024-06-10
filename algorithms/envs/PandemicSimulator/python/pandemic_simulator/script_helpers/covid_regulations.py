# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import List

from ..environment import PandemicRegulation, DEFAULT, Risk, Office, School, HairSalon, RetailStore, Restaurant, Bar

__all__ = ['austin_regulations', 'italian_regulations', 'swedish_regulations']

austin_regulations: List[PandemicRegulation] = [
    PandemicRegulation(stay_home_if_sick=False,
                       practice_good_hygiene=False,
                       wear_facial_coverings=False,
                       social_distancing=DEFAULT,
                       risk_to_avoid_gathering_size={Risk.LOW: -1, Risk.HIGH: -1},
                       location_type_to_rule_kwargs={
                           Office: {'lock': False},
                           School: {'lock': False},
                           HairSalon: {'lock': False},
                           RetailStore: {'lock': False},
                           Bar: {'lock': False},
                           Restaurant: {'lock': False},
                       },
                       stage=0),
    PandemicRegulation(stay_home_if_sick=True,
                       practice_good_hygiene=True,
                       wear_facial_coverings=False,
                       social_distancing=DEFAULT,
                       risk_to_avoid_gathering_size={Risk.HIGH: 25, Risk.LOW: 50},
                       location_type_to_rule_kwargs={
                           Office: {'lock': False},
                           School: {'lock': False},
                           HairSalon: {'lock': False},
                           RetailStore: {'lock': False},
                           Restaurant: {'lock': False},
                           Bar: {'lock': False},
                       },
                       stage=1),
    PandemicRegulation(stay_home_if_sick=True,
                       practice_good_hygiene=True,
                       wear_facial_coverings=True,
                       social_distancing=0.3,
                       risk_to_avoid_gathering_size={Risk.HIGH: 10, Risk.LOW: 25},
                       location_type_to_rule_kwargs={
                           Office: {'lock': False},
                           School: {'lock': True},
                           HairSalon: {'lock': True},
                           RetailStore: {'lock': False},
                           Restaurant: {'lock': False},
                           Bar: {'lock': False},
                       },
                       stage=2),
    PandemicRegulation(stay_home_if_sick=True,
                       practice_good_hygiene=True,
                       wear_facial_coverings=True,
                       social_distancing=0.5,
                       risk_to_avoid_gathering_size={Risk.HIGH: 0, Risk.LOW: 0},
                       location_type_to_rule_kwargs={
                           Office: {'lock': False},
                           School: {'lock': True},
                           HairSalon: {'lock': True},
                           RetailStore: {'lock': False},
                           Restaurant: {'lock': True},
                           Bar: {'lock': True},
                       },
                       stage=3),
    PandemicRegulation(stay_home_if_sick=True,
                       practice_good_hygiene=True,
                       wear_facial_coverings=True,
                       social_distancing=0.7,
                       risk_to_avoid_gathering_size={Risk.HIGH: 0, Risk.LOW: 0},
                       location_type_to_rule_kwargs={
                           Office: {'lock': True},
                           School: {'lock': True},
                           HairSalon: {'lock': True},
                           RetailStore: {'lock': True},
                           Restaurant: {'lock': True},
                           Bar: {'lock': True},
                       },
                       stage=4)
]

# https://en.wikipedia.org/wiki/COVID-19_pandemic_in_Italy#First_measures

# STAGE 0
# January 31 first cases

# STAGE 1
# On 1 March, the Council of Ministers approved a decree to organise the containment of the outbreak. ...
# The rest of the national territory, where safety and prevention measures are advertised in public places and special
# sanitisations are performed on means of public transport.

# STAGE 2
# On 4 March, the Italian government imposed the shutdown of all schools and universities nationwide for two weeks

# STAGE 3
# On 9 March, ... In the evening, Conte announced in a press conference that all measures previously applied only in
# the so-called "red zones" had been extended to the whole country, putting approximately 60 million people in lockdown.
# The decree also established the closure of all gyms, swimming pools, spas and wellness centres. Shopping centres had
# to be closed on weekends, while other commercial activities could remain open if a distance of one metre between
# customers could be guaranteed.
# The decree imposed the closure of museums, cultural centres and ski resorts ... and the closure of cinemas, theatres,
# pubs, dance schools, game rooms, betting rooms and bingo halls, discos and similar places in the entire country.
# Civil and religious ceremonies, including funeral ceremonies, were suspended. All organised events were also
# suspended, as well as events in public or private places
# On 11 March, .. In the evening, Conte announced a tightening of the lockdown,
# with all commercial and retail businesses except those providing essential services, like
# grocery shops and pharmacies, closed down.

# STAGE 4
# On 21 March, Conte announced further restrictions within the nationwide lockdown, by halting all non-essential
# production, industries and businesses in Italy, following the rise in the number of new cases and deaths in the
# previous days.

# BACK TO STAGE 3
# On 26 April, the Prime Minister announced a starter plan for the so-called "phase 2", that would start from 4 May.
# Movements across regions would still be forbidden, while the ones between municipalities would be allowed only for
# work and health reasons, as well as for visits to relatives. The plan allowed the re-opening of manufacturing
# industries and construction sites, however schools, bars, restaurants and hairdressers would stay closed

# BACK TO STAGE 2
# On 13 May, Education Minister Lucia Azzolina announced schools would remain closed until September.
# On 16 May, Conte announced the government plan for the easing of restrictions. Starting from 18 May most businesses
# could reopen, and free movement was granted to all citizens within their Region; movement across Regions was still
# banned for non-essential motives. Furthermore, on 25 May swimming pools and gyms could also reopen, and on 15 June
# theatres and cinemas.
italian_regulations: List[PandemicRegulation] = [
    PandemicRegulation(stay_home_if_sick=False,
                       practice_good_hygiene=False,
                       wear_facial_coverings=False,
                       social_distancing=DEFAULT,
                       risk_to_avoid_gathering_size={Risk.LOW: -1, Risk.HIGH: -1},
                       location_type_to_rule_kwargs={
                           Office: {'lock': False},
                           School: {'lock': False},
                           HairSalon: {'lock': False},
                           RetailStore: {'lock': False},
                           Bar: {'lock': False},
                           Restaurant: {'lock': False},
                       },
                       stage=0),
    PandemicRegulation(stay_home_if_sick=True,
                       practice_good_hygiene=True,
                       wear_facial_coverings=False,
                       social_distancing=0.2,
                       location_type_to_rule_kwargs={
                           Office: {'lock': False},
                           School: {'lock': False},
                           HairSalon: {'lock': False},
                           RetailStore: {'lock': False},
                           Restaurant: {'lock': False},
                           Bar: {'lock': False},

                       },
                       stage=1),
    PandemicRegulation(stay_home_if_sick=True,
                       practice_good_hygiene=True,
                       wear_facial_coverings=False,
                       social_distancing=0.25,
                       location_type_to_rule_kwargs={
                           Office: {'lock': False},
                           School: {'lock': True},
                           HairSalon: {'lock': False},
                           RetailStore: {'lock': False},
                           Restaurant: {'lock': False},
                           Bar: {'lock': False},
                       },
                       stage=2),
    PandemicRegulation(stay_home_if_sick=True,
                       practice_good_hygiene=True,
                       wear_facial_coverings=True,
                       social_distancing=0.6,
                       risk_to_avoid_gathering_size={Risk.HIGH: 0, Risk.LOW: 0},
                       location_type_to_rule_kwargs={
                           Office: {'lock': False},
                           School: {'lock': True},
                           HairSalon: {'lock': True},
                           RetailStore: {'lock': True},
                           Restaurant: {'lock': True},
                           Bar: {'lock': True},
                       },
                       stage=3),
    PandemicRegulation(stay_home_if_sick=True,
                       practice_good_hygiene=True,
                       wear_facial_coverings=True,
                       social_distancing=0.8,
                       risk_to_avoid_gathering_size={Risk.HIGH: 0, Risk.LOW: 0},
                       location_type_to_rule_kwargs={
                           Office: {'lock': True},
                           School: {'lock': True},
                           HairSalon: {'lock': True},
                           RetailStore: {'lock': True},
                           Restaurant: {'lock': True},
                           Bar: {'lock': True},
                       },
                       stage=4)
]

# https://home.kpmg/xx/en/home/insights/2020/04/sweden-government-and-institution-measures-in-response-to-covid.html
# Sweden took no nationwide lockdown; Remote work *recommended*;
# Schools are open; Restaurants are open.
# Travel ban.

# https://www.folkhalsomyndigheten.se/the-public-health-agency-of-sweden/communicable-disease-control/covid-19/prevention/
# We do not currently recommend face masks in public settings since the scientific evidence
# around the effectiveness of face masks in combatting the spread of infection is unclear.
# https://www.folkhalsomyndigheten.se/the-public-health-agency-of-sweden/communicable-disease-control/
# covid-19--the-swedish-strategy/

# Anders Tegnell says his modelling indicates that, on average, Swedes have around 30% of the social interactions they
# did prior to the pandemic.

swedish_regulations: List[PandemicRegulation] = [
    PandemicRegulation(stay_home_if_sick=False,
                       practice_good_hygiene=False,
                       wear_facial_coverings=False,
                       social_distancing=DEFAULT,
                       risk_to_avoid_gathering_size={Risk.LOW: -1, Risk.HIGH: -1},
                       location_type_to_rule_kwargs={
                           Office: {'lock': False},
                           School: {'lock': False},
                           HairSalon: {'lock': False},
                           RetailStore: {'lock': False},
                           Bar: {'lock': False},
                           Restaurant: {'lock': False},
                       },
                       stage=0),
    PandemicRegulation(stay_home_if_sick=True,
                       practice_good_hygiene=True,
                       social_distancing=0.139,  # after calibration
                       risk_to_avoid_gathering_size={Risk.HIGH: 50, Risk.LOW: 50},
                       stage=1),
]
