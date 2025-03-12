# %% [markdown]
# ### Imports

# %%
# Importing API libraries
from dotenv import load_dotenv
import os
import requests

# Handling API limits libraries
from concurrent.futures import as_completed, ProcessPoolExecutor
from requests_futures.sessions import FuturesSession
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# SQL imports
from pangres import upsert
from sqlalchemy import text, create_engine

# Stanadard libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

# %% [markdown]
# ### Loading Data Into SQL Database Functions

# %%
# loading env information
db_username = os.environ.get('db_username')
db_password = os.environ.get('db_password')
db_host = os.environ.get('db_host')
db_port = os.environ.get('db_port')
db_name = os.environ.get('db_name')

# creating db url
def create_db_connection_str(db_username, db_password, db_host, db_port, db_name):
    connection_url = 'postgresql+psycopg2://'+db_username+':'+db_password+'@'+db_host+':'+db_port+'/'+db_name
    return connection_url

# creating db engine
def create_db_engine(db_username, db_password, db_host, db_port, db_name):
    db_url = create_db_connection_str(db_username, db_password, db_host, db_port, db_name)
    db_engine = create_engine(db_url, pool_recycle=3600, future=True)

    return db_engine

def upload_data_into_sql(dataframe,neededSchema,tablename):
    db_engine = create_db_engine(db_username, db_password, db_host, db_port, db_name)
    connection = db_engine.connect()
    upsert(con=connection,df=dataframe, schema=neededSchema, table_name=tablename, create_table=True,create_schema=True, if_row_exists='update')
    connection.commit()

def load_SQL_Data():
    db_engine = create_db_engine(db_username, db_password, db_host, db_port, db_name)
    with db_engine.connect() as connection:
        df = pd.read_sql(text('select * from soloq.top_ladder_matches'),connection)
    df = df.reset_index(drop=True)
    return df 

def get_stored_matches():
    df = load_SQL_Data()
    return df['matchId']


# %% [markdown]
# ### Obtaining Puuid Functions

# %%
# Loading dotenv file for code
load_dotenv()

# Riot Games API Key
API_Key = os.environ.get('riot_api_key')

# Global counter to track how many times the function is run
puuid_call_count = 0  # Initialize the counter to 0

# getting puuids
def get_puuid(SummonerId=None, PlayerName=None, PlayerTag=None, max_retries=5):
    """Fetches the PUUID of a player using either Summoner ID or Riot ID (PlayerName + PlayerTag)."""
    
    global puuid_call_count  # Use the global counter
    
    puuid_call_count += 1  # Increment counter every time the function is called
    
    retries = 0  # Track retry attempts

    while retries < max_retries:
        if SummonerId is not None:
            root_url = 'https://na1.api.riotgames.com'  # NA1 region for SummonerId lookup
            endpoint = f'/lol/summoner/v4/summoners/{SummonerId}'
        else:
            root_url = 'https://americas.api.riotgames.com'  # Americas region for Riot ID lookup
            endpoint = f'/riot/account/v1/accounts/by-riot-id/{PlayerName}/{PlayerTag}'

        response = requests.get(root_url + endpoint + "?api_key=" + API_Key)

        # Debugging: Show response code and function call count
        print(f"PUUID Function Call #{puuid_call_count} - URL: {root_url}{endpoint}, Response Code: {response.status_code}")

        if response.status_code == 200:
            try:
                return response.json()['puuid'], 200  # Return PUUID and success code
            except KeyError:
                print(f"Error: Missing 'puuid' in API response for {SummonerId or PlayerName}#{PlayerTag}")
                return None, 500  # Return None if JSON response doesn't have expected data

        elif response.status_code == 429:
            print(f"Rate limited on PUUID request! Waiting for 2 minutes 5 seconds before retrying ({retries+1}/{max_retries})...")
            time.sleep(125)  # Wait before retrying
            retries += 1  # Increment retry count
        else:
            print(f"Failed to retrieve PUUID. Status Code: {response.status_code}")
            return None, response.status_code  # Stop retrying on other errors

    print(f"Max retries reached for {SummonerId or PlayerName}#{PlayerTag}, skipping PUUID retrieval.")
    return None, 429  # Return None after max retries

  

#print(get_puuid(PlayerName='BoopThySnoot',PlayerTag='NA1'))

# Getting player name and tag from puuid
def get_idtag_from_puuid(puuid=None, region = 'americas'):
  # The region for this function is americas
  root_url = f'https://{region}.api.riotgames.com'
  endpoint = f'/riot/account/v1/accounts/by-puuid/{puuid}'
  # root and endpoint makes creating the links easier
  response = requests.get(root_url + endpoint + "?api_key=" + API_Key)

  # combining tag and name to be like a standard riot id
  id = {
    'playerName' : response.json()['gameName'],
    'playerTag' : response.json()['tagLine']
  }
  return id


#print(get_idtag_from_puuid(get_puuid(PlayerName='BoopThySnoot',PlayerTag='NA1'))['playerName'])

# %% [markdown]
# ### Obtaining Ladder Function

# %%
# Obtaining Challanger & Masters leagues ranked ladder

def get_ladder(top = None, region = 'na1'):
    
    # root for all leagues 
    root = f'https://{region}.api.riotgames.com'
    # differnet leagues endpoints
    chall_endpoint = '/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5'
    gm_endpoint = '/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5'
    masters_endpoint = '/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5'

    #challenger df
    response = requests.get(root + chall_endpoint + "?api_key=" + API_Key)
    chall_df = pd.DataFrame(response.json()['entries']).sort_values('leaguePoints', ascending=False).reset_index(drop = True)
    chall_df

    # other dfs
    gm_df = pd.DataFrame()
    masters_df = pd.DataFrame()

    # Creating ladder df (only pulls other ladders if pull size requires)

    if top > 300: 
        response = requests.get(root + gm_endpoint + "?api_key=" + API_Key)
        gm_df = pd.DataFrame(response.json()['entries']).sort_values('leaguePoints', ascending=False).reset_index(drop = True)
    ladder = pd.concat([chall_df,gm_df,masters_df]).reset_index(drop=True)[:top] # selects x amount of players
    if top > 1000:
        response = requests.get(root + masters_endpoint + "?api_key=" + API_Key)
        masters_df = pd.DataFrame(response.json()['entries']).sort_values('leaguePoints', ascending=False).reset_index(drop = True)

    #creating representative rank column
    ladder = ladder.reset_index(drop=False).drop(columns='rank').rename(columns={'index':'rank'})
    ladder['rank'] = ladder['rank'] + 1

    #output
    return ladder

get_ladder(top = 1500)

# %% [markdown]
# ### Obtaining Match History Functions

# %%
# Obtaining match histories 
def get_match_hist(puuid = None, start = 0, count = 20, region = 'americas', removestored = False):

  # Set up a session with retry mechanisms
  session = FuturesSession(executor=ProcessPoolExecutor(max_workers=10))
  retries = 5
  status_forcelist = [429]
  retry = Retry(
      total=retries,
      read=retries,
      connect=retries,
      respect_retry_after_header=True,
      status_forcelist=status_forcelist,
  )

  adapter = HTTPAdapter(max_retries=retry)
  session.mount('http://', adapter)
  session.mount('https://', adapter)

# Building the Url
  root_url = f'https://{region}.api.riotgames.com'
  endpoint = f'/lol/match/v5/matches/by-puuid/{puuid}/ids'
  query_params = f'?type=ranked&start={start}&count={count}'
  response = requests.get(root_url + endpoint + query_params + "&api_key=" + API_Key)
  matchIds = response.json()

  return matchIds

#print(get_match_hist(get_puuid(SummonerId= 'uWLwLzOWUUAe2fDmPwTQEUm0_JisdwLXqsrUM-SDb64OyEE'),count=10))

# getting the data from the match histories 
def get_match_data_from_id(matchId = None, region = 'americas'):

    # Set up a session with retry mechanisms
  session = FuturesSession(executor=ProcessPoolExecutor(max_workers=10))
  retries = 5
  status_forcelist = [429]
  retry = Retry(
      total=retries,
      read=retries,
      connect=retries,
      respect_retry_after_header=True,
      status_forcelist=status_forcelist,
  )

  adapter = HTTPAdapter(max_retries=retry)
  session.mount('http://', adapter)
  session.mount('https://', adapter)

  # building the url
  root_url = f'https://{region}.api.riotgames.com'
  endpoint = f'/lol/match/v5/matches/{matchId}'

  response = requests.get(root_url + endpoint + "?api_key=" + API_Key)
  # Debugging: Show response code
  print(f"Match ID: {matchId}, Response Code: {response.status_code}")

  try:
      return response.json(), response.status_code  # Return both JSON and status code
  except:
      return None, response.status_code  # Return None if JSON parsing fails



#get_match_data_from_id(matchId = 'NA1_5218927642')



# %% [markdown]
# ### Processing Match History Function

# %%
# Data Processing Setup 
#game = get_match_data_from_id(matchId = 'NA1_5218927642')

def process_match_json(match_json, puuid):
    
    try: 
        # match_json should include the get_match_data_from_id() function like above ^
        # All time units are in seconds


        # Architecture 
        metadata = match_json['metadata']
        info = match_json['info']
        participants = metadata['participants']
        players = info['participants']
        player = players[participants.index(puuid)]
        teams = info['teams']
        player_team_id = player['teamId'] #100 is blueside and 200 is redside
        player_team = next((team for team in teams if team['teamId'] == player_team_id), None)
        enemy_team = next((team for team in teams if team['teamId'] != player_team_id), None)
        obj_team = player_team['objectives']
        try:
            atakhan_team = obj_team['atakhan']
        except: 
            atakhan_team = None
        baron_team = obj_team['baron']
        dragon_team = obj_team['dragon']
        voidGrubs_team = obj_team['horde']
        riftHerald_team = obj_team['riftHerald']
        perks = player['perks']
        stats = perks['statPerks']
        styles = perks['styles']
        primary = styles[0]
        secondary = styles[1]

        # Match metadata
        matchId = metadata['matchId']
        participants = metadata['participants']

        # Match information
        endOfGameResult = info['endOfGameResult'] # this needs to return 'GameComplete'
        gameCreation = info['gameCreation']
        gameDuration = info['gameDuration']
        gameEndTimestamp =  info['gameEndTimestamp']
        gameId = info['gameId']
        patch = info['gameVersion']

        # player information
        riotIdGameName = player['riotIdGameName']
        riotIdTagline = player['riotIdTagline']
        summonerName = player['summonerName']

        # kda information
        kills = player['kills']
        deaths = player['deaths']
        assists = player['assists']
        firstBloodKill = player['firstBloodKill'] 
        firstBloodAssist = player['firstBloodAssist']
        doubleKills = player['doubleKills']
        tripleKills = player['tripleKills']
        quadraKills = player['quadraKills']
        pentaKills = player['pentaKills']
        unrealKills = player['unrealKills']
        largestMultiKill = player['largestMultiKill']
        killingSprees = player['killingSprees']
        largestKillingSpree = player['largestKillingSpree']
        teamKills = player_team['objectives']['champion']['kills']

        # Time information
        timePlayed = player['timePlayed']
        longestTimeSpentLiving = player['longestTimeSpentLiving']
        totalTimeSpentDead = player['totalTimeSpentDead']
        perctTimeDead = round((totalTimeSpentDead / timePlayed) * 100, 2)

        # Minion kill info
        cs = player['totalMinionsKilled']
        neutralMinionsKilled = player['neutralMinionsKilled']
        totalAllyJungleMinionsKilled = player['totalAllyJungleMinionsKilled']
        totalEnemyJungleMinionsKilled = player['totalEnemyJungleMinionsKilled']

        # Ping info
        OMWPing = player['onMyWayPings']
        allInPings = player['allInPings']
        assistMePings = player['assistMePings']
        enemyMissingPings = player['enemyMissingPings']
        enemyVisionPings = player['enemyVisionPings']
        holdPings = player['holdPings']
        getBackPings = player['getBackPings']
        needVisionPings = player['needVisionPings']
        pushPings = player['pushPings']

        # vision info
        visionScore = player['visionScore']
        visionClearedPings = player['visionClearedPings']
        visionWardsBoughtInGame = player['visionWardsBoughtInGame']
        wardsKilled = player['wardsKilled']
        wardsPlaced = player['wardsPlaced']
        sightWardsBoughtInGame = player['sightWardsBoughtInGame']
        detectorWardsPlaced = player['detectorWardsPlaced']

        # Major objective info
        baronKills = player['baronKills']
        dragonKills = player['dragonKills']
        damageDealtToObjectives = player['damageDealtToObjectives']
        objectivesStolen = player['objectivesStolen']
        objectivesStolenAssists = player['objectivesStolenAssists']

        # turret info
        turretKills = player['turretKills']
        turretTakedowns = player['turretTakedowns']
        turretsLost = player['turretsLost']
        firstTowerAssist = player['firstTowerAssist']
        firstTowerKill = player['firstTowerKill']
        damageDealtToBuildings = player['damageDealtToBuildings']
        damageDealtToTurrets = player['damageDealtToTurrets']

        # inhib info
        inhibitorKills = player['inhibitorKills']
        inhibitorTakedowns = player['inhibitorTakedowns']
        inhibitorsLost = player['inhibitorsLost']

        # game position
        teamPosition = player['teamPosition']
        lane = player['lane']
        role = player['role']

        # champion info
        championName = player['championName']
        champExperience = player['champExperience']
        champLevel = player['champLevel']
        championTransform = player['championTransform']

        # Perk info
        primary_style = primary['style']
        secondary_style = secondary['style']

        primary_keystone = primary['selections'][0]['perk']
        primary_perk_1 = primary['selections'][1]['perk']
        primary_perk_2 = primary['selections'][2]['perk']
        primary_perk_3 = primary['selections'][3]['perk']

        secondary_perk_1 = secondary['selections'][0]['perk']
        secondary_perk_2 = secondary['selections'][1]['perk']

        # summoner spells
        summoner1Id = player['summoner1Id']
        summoner2Id = player['summoner2Id']

        # Champion Stat info
        defense = stats['defense']
        flex = stats['flex']
        offense = stats['offense']

        # gold info
        goldEarned = player['goldEarned']
        goldSpent = player['goldSpent']
        bountyLevel = player['bountyLevel']

        # item info
        item0 = player['item0']
        item1 = player['item1']
        item2 = player['item2']
        item3 = player['item3']
        item4 = player['item4']
        item5 = player['item5']
        item6 = player['item6']
        itemsPurchased = player['itemsPurchased']
        consumablesPurchased = player['consumablesPurchased']

        # Interaction info
        totalDamageDealt = player['totalDamageDealt']
        magicDamageDealt = player['magicDamageDealt']
        physicalDamageDealt = player['physicalDamageDealt']
        trueDamageDealt = player['trueDamageDealt']

        totalDamageDealtToChampions = player['totalDamageDealtToChampions']
        magicDamageDealtToChampions = player['magicDamageDealtToChampions']
        physicalDamageDealtToChampions = player['physicalDamageDealtToChampions']
        trueDamageDealtToChampions = player['trueDamageDealtToChampions']

        totalDamageTaken = player['totalDamageTaken']
        magicDamageTaken = player['magicDamageTaken']
        physicalDamageTaken = player['physicalDamageTaken']
        trueDamageTaken = player['trueDamageTaken']

        totalHeal = player['totalHeal']
        totalUnitsHealed = player['totalUnitsHealed']
        totalHealsOnTeammates = player['totalHealsOnTeammates']
        totalDamageShieldedOnTeammates = player['totalDamageShieldedOnTeammates']
        damageSelfMitigated = player['damageSelfMitigated']

        timeCCingOthers = player['timeCCingOthers']
        totalTimeCCDealt = player['totalTimeCCDealt']

        # team objectives
        if atakhan_team is not None:
            atakhan_killed = atakhan_team['first']
        else:
            atakhan_killed =  'False'
        baron_first = baron_team['first']
        baron_kills = baron_team['kills']
        dragon_first = dragon_team['first']
        dragon_kills = dragon_team['kills']
        voidGrubs_first = voidGrubs_team['first']
        voidGrubs_kills = voidGrubs_team['kills']
        riftHerald_first =  riftHerald_team['first']
        riftHerald_kills = riftHerald_team['kills']

        # Team champion info 
        # Initialize lists for blue and red side champions
        blue_side_champions = []
        red_side_champions = []

        # Loop through all players and categorize champions by team
        for playerr in players:
            player_Champion = playerr['championName']
            team_id = playerr['teamId']  # Get player's team ID
            
            if team_id == 100:
                blue_side_champions.append(player_Champion)
            elif team_id == 200:
                red_side_champions.append(player_Champion)

        # Create a dictionary separating blue and red side champions
        champion_dict = {
            "Blue_Side": blue_side_champions,
            "Red_Side": red_side_champions
        }

        # Assign blue side champions to individual variables
        blue_champ1, blue_champ2, blue_champ3, blue_champ4, blue_champ5 = blue_side_champions

        # Assign red side champions to individual variables
        red_champ1, red_champ2, red_champ3, red_champ4, red_champ5 = red_side_champions

        # Game outcome
        win = player['win']
        
        # creating dataframe from match json
        matchDF = pd.DataFrame({
            'matchId': [matchId],
            'participants': [participants],
            'puuid' : [puuid],
            'endOfGameResult': [endOfGameResult],
            'gameCreation': [gameCreation],
            'gameDuration': [gameDuration],
            'gameEndTimestamp': [gameEndTimestamp],
            'gameId': [gameId],
            'patch': [patch],
            'riotIdGameName': [riotIdGameName],
            'riotIdTagline': [riotIdTagline],
            'summonerName': [summonerName],
            'kills': [kills],
            'deaths': [deaths],
            'assists': [assists],
            'firstBloodKill': [firstBloodKill],
            'firstBloodAssist': [firstBloodAssist],
            'doubleKills': [doubleKills],
            'tripleKills': [tripleKills],
            'quadraKills': [quadraKills],
            'pentaKills': [pentaKills],
            'unrealKills': [unrealKills],
            'largestMultiKill': [largestMultiKill],
            'killingSprees': [killingSprees],
            'largestKillingSpree': [largestKillingSpree],
            'teamKills': [teamKills],
            'timePlayed': [timePlayed],
            'longestTimeSpentLiving': [longestTimeSpentLiving],
            'totalTimeSpentDead': [totalTimeSpentDead],
            'perctTimeDead': [perctTimeDead],
            'cs': [cs],
            'neutralMinionsKilled': [neutralMinionsKilled],
            'totalAllyJungleMinionsKilled': [totalAllyJungleMinionsKilled],
            'totalEnemyJungleMinionsKilled': [totalEnemyJungleMinionsKilled],
            'OMWPing': [OMWPing],
            'allInPings': [allInPings],
            'assistMePings': [assistMePings],
            'enemyMissingPings': [enemyMissingPings],
            'enemyVisionPings': [enemyVisionPings],
            'holdPings': [holdPings],
            'getBackPings': [getBackPings],
            'needVisionPings': [needVisionPings],
            'pushPings': [pushPings],
            'visionScore': [visionScore],
            'visionClearedPings': [visionClearedPings],
            'visionWardsBoughtInGame': [visionWardsBoughtInGame],
            'wardsKilled': [wardsKilled],
            'wardsPlaced': [wardsPlaced],
            'sightWardsBoughtInGame': [sightWardsBoughtInGame],
            'detectorWardsPlaced': [detectorWardsPlaced],
            'baronKills': [baronKills],
            'dragonKills': [dragonKills],
            'damageDealtToObjectives': [damageDealtToObjectives],
            'objectivesStolen': [objectivesStolen],
            'objectivesStolenAssists': [objectivesStolenAssists],
            'turretKills': [turretKills],
            'turretTakedowns': [turretTakedowns],
            'turretsLost': [turretsLost],
            'firstTowerAssist': [firstTowerAssist],
            'firstTowerKill': [firstTowerKill],
            'damageDealtToBuildings': [damageDealtToBuildings],
            'damageDealtToTurrets': [damageDealtToTurrets],
            'inhibitorKills': [inhibitorKills],
            'inhibitorTakedowns': [inhibitorTakedowns],
            'inhibitorsLost': [inhibitorsLost],
            'teamPosition': [teamPosition],
            'lane': [lane],
            'role': [role],
            'championName': [championName],
            'champExperience': [champExperience],
            'champLevel': [champLevel],
            'championTransform': [championTransform],
            'primary_style': [primary_style],
            'secondary_style': [secondary_style],
            'primary_keystone': [primary_keystone],
            'primary_perk_1': [primary_perk_1],
            'primary_perk_2': [primary_perk_2],
            'primary_perk_3': [primary_perk_3],
            'secondary_perk_1': [secondary_perk_1],
            'secondary_perk_2': [secondary_perk_2],
            'summoner1Id': [summoner1Id],
            'summoner2Id': [summoner2Id],
            'defense': [defense],
            'flex': [flex],
            'offense': [offense],
            'goldEarned': [goldEarned],
            'goldSpent': [goldSpent],
            'bountyLevel': [bountyLevel],
            'item0': [item0],
            'item1': [item1],
            'item2': [item2],
            'item3': [item3],
            'item4': [item4],
            'item5': [item5],
            'wardTypeBought': [item6],
            'itemsPurchased': [itemsPurchased],
            'consumablesPurchased': [consumablesPurchased],
            'totalDamageDealt': [totalDamageDealt],
            'magicDamageDealt': [magicDamageDealt],
            'physicalDamageDealt': [physicalDamageDealt],
            'trueDamageDealt': [trueDamageDealt],
            'totalDamageDealtToChampions': [totalDamageDealtToChampions],
            'magicDamageDealtToChampions': [magicDamageDealtToChampions],
            'physicalDamageDealtToChampions': [physicalDamageDealtToChampions],
            'trueDamageDealtToChampions': [trueDamageDealtToChampions],
            'totalDamageTaken': [totalDamageTaken],
            'magicDamageTaken': [magicDamageTaken],
            'physicalDamageTaken': [physicalDamageTaken],
            'trueDamageTaken': [trueDamageTaken],
            'totalHeal': [totalHeal],
            'totalUnitsHealed': [totalUnitsHealed],
            'totalHealsOnTeammates': [totalHealsOnTeammates],
            'totalDamageShieldedOnTeammates': [totalDamageShieldedOnTeammates],
            'damageSelfMitigated': [damageSelfMitigated],
            'timeCCingOthers': [timeCCingOthers],
            'totalTimeCCDealt': [totalTimeCCDealt],
            'atakhan_killed': [atakhan_killed],
            'playerSide' : [player_team_id],
            'blue_champ1' : [blue_champ1],
            'blue_champ2' : [blue_champ2],
            'blue_champ3' : [blue_champ3],
            'blue_champ4' : [blue_champ4],
            'blue_champ5' : [blue_champ5],
            'red_champ1' : [red_champ1],
            'red_champ2' : [red_champ2],
            'red_champ3' : [red_champ3],
            'red_champ4' : [red_champ4],
            'red_champ5' : [red_champ5],
            'baron_first': [baron_first],
            'baron_kills': [baron_kills],
            'dragon_first': [dragon_first],
            'dragon_kills': [dragon_kills],
            'voidGrubs_first': [voidGrubs_first],
            'voidGrubs_kills': [voidGrubs_kills],
            'riftHerald_first': [riftHerald_first],
            'riftHerald_kills': [riftHerald_kills],
            'win': [win]
        })

        return matchDF
    except:
        return pd.DataFrame()


# %% [markdown]
# ### Making Values Readable Functions

# %%
def convert_df_ids(df):   
 # Url links to get the names for each of the ids
    champ_name = 'https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-summary.json'
    perks = 'https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/perks.json'
    perkstyles = 'https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/perkstyles.json'
    summoner_spells = 'https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/summoner-spells.json'
    item_names = 'https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/items.json'

    # converting ids to json 
    champ_name_json = requests.get(champ_name).json()
    perks_json = requests.get(perks).json()
    perkstyles_json = requests.get(perkstyles).json()
    summoner_spells_json = requests.get(summoner_spells).json()
    item_names_json =  requests.get(item_names).json()

    # Function to extract id-name pairs from each JSON
    def extract_id_name(data):
        # Initialize an empty dictionary
        id_name_dict = {}
        
        # Loop through each item in the list (or adjust based on structure)
        if isinstance(data, list):
            for item in data:
                # Assuming each item has 'id' and 'name' keys
                id_name_dict[item['id']] = item['name']
        elif isinstance(data, dict):
            # Handle the case where data is a dictionary
            for key, item in data.items():
                if isinstance(item, list):
                    for sub_item in item:
                        id_name_dict[sub_item['id']] = sub_item['name']
                        
        return id_name_dict

    # Extract id-name pairs for each JSON
    champ_name_dict = extract_id_name(champ_name_json)
    perks_dict = extract_id_name(perks_json)
    perkstyles_dict = extract_id_name(perkstyles_json)
    summoner_spells_dict = extract_id_name(summoner_spells_json)
    item_names_dict = extract_id_name(item_names_json)

    # replacing values into data frame
    df['championName'] = df['championName'].replace(champ_name_dict)
    df[['primary_style','secondary_style']] = df[['primary_style','secondary_style']].replace(perkstyles_dict)
    df[['primary_keystone','primary_perk_1','primary_perk_2','primary_perk_3','secondary_perk_1','secondary_perk_2']]= df[['primary_keystone','primary_perk_1','primary_perk_2','primary_perk_3','secondary_perk_1','secondary_perk_2']].replace(perks_dict)
    df[['summoner1Id','summoner2Id']] = df[['summoner1Id','summoner2Id']].replace(summoner_spells_dict)
    df[['item0','item1','item2','item3','item4','item5','wardTypeBought']] = df[['item0','item1','item2','item3','item4','item5','wardTypeBought']].replace(item_names_dict)

    # replacing player side with text
    df['playerSide'] = df['playerSide'].replace({100: 'Blue', 200: 'Red'})

    return df

# %% [markdown]
# ### Obtaining Match Data Function

# %%
def get_data(players_wanted=None, ladder_start=None, match_count=None):
    # Get ladder data 
    ladder = get_ladder(players_wanted+ladder_start)
    ladder = pd.DataFrame(ladder['summonerId'][ladder_start:])
    summonerIds = ladder['summonerId'].to_list()

    # Convert summoner IDs to PUUIDs with rate-limiting handling
    puuids = {}
    for sumid in summonerIds:
        while True:  # Keep retrying if status code is 429
            puuid, status_code = get_puuid(sumid)

            if status_code == 200:
                puuids[sumid] = puuid  # Store valid PUUID
                break  # Exit loop on success
            elif status_code == 429:
                print(f"Rate limited on PUUID request! Waiting for 2 minutes 5 seconds before retrying summoner {sumid}...")
                time.sleep(125)  # Wait for 2 minutes  5 seconds before retrying
            else:
                print(f"Failed to retrieve PUUID for summoner {sumid}. Status Code: {status_code}")
                break  # Stop retrying on other errors

    # Get match histories and store them in a dictionary
    match_dict = {}  # {match_id: [puuid1, puuid2, ...]}
    for summoner_id, puuid in puuids.items():
        match_ids = get_match_hist(puuid, count=match_count)
        for match_id in match_ids:
            if match_id not in match_dict:
                match_dict[match_id] = []  # Initialize list if match_id is new
            match_dict[match_id].append(puuid)  # Append PUUID instead of overwriting

    # List to collect DataFrames
    dataframes = []

    # Loop through matches and collect DataFrames
    for match_id, puid_list in match_dict.items():
        while True:  # Keep retrying if status code is 429
            game, status_code = get_match_data_from_id(matchId=match_id)
            
            if status_code == 200:
                break  # Valid response, exit loop
            elif status_code == 429:
                print(f"Rate limited on match data! Waiting for 2 minutes 5 seconds before retrying match {match_id}...")
                time.sleep(125)  # Wait for 2 minutes 5 seconds before retrying
            else:
                print(f"Failed to retrieve match {match_id}. Status Code: {status_code}")
                break  # Stop retrying on other errors

        if game and status_code == 200:
            for puid in puid_list:  # Process for each PUUID in the match
                matchDF = process_match_json(game, puuid=puid)

                print(f"Processing Match ID: {match_id}, PUUID: {puid}, DataFrame Shape: {matchDF.shape}")  # Debugging print
                matchDF = convert_df_ids(matchDF)
                matchDF['uuid'] = matchDF['matchId'] + '_' + matchDF['puuid']
                matchDF = matchDF.set_index('uuid')
                upload_data_into_sql(matchDF, 'soloq', 'top_ladder_matches')
                print(f"Match: {match_id} has been fully processed and uploaded to SQL")

                # Ensure matchDF is not empty before appending
                if not matchDF.empty:
                    dataframes.append(matchDF)
                    print(f'There are currently {len(dataframes)} columns!, processed {round(len(dataframes)/25,2)} players. ')

    # Concatenate all dataframes
    df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

    return df


# %% [markdown]
# ### Converting Data to CSV

# %%
# get the data set
#get_data(players_wanted = 100, ladder_start=349,  match_count=25)

# %%
# Checking that the data was loading into the frame
#df = load_SQL_Data()
#df.shape

# saving the data into a csv file
#df.to_csv('C:\\Users\\Kadin\\Desktop\\DS Capstone\\LOLChallengerData.csv', index=False)


