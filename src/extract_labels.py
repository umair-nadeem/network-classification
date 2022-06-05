import os
import itertools
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.style.use('seaborn')
os.chdir('/home/umair/PycharmProjects/thesis/files/classification_csv')

final_df = pd.read_csv('flows.csv')

labels = list(set(final_df['appName']))
data = np.asarray([])
size = np.asarray([])

final_labels = ['ICMP', 'ICMP', 'DNS', 'FTP',

                'SMTP', 'POP', 'IMAP',

                'SSH',

                'HTTPImageTransfer', 'HTTPWeb',

                'WebFileTransfer', 'WindowsFileSharing',

                'WebMediaVideo', 'WebMediaAudio', 'WebMediaDocuments',

                'Unknown_TCP', 'Unknown_UDP', 'MiscApplication',

                'BitTorrent', 'PeerEnabler']

final_df = final_df[final_df['appName'].isin(final_labels)]
final_df['startDateTime'] = pd.to_datetime(final_df['startDateTime'])
final_df['stopDateTime'] = pd.to_datetime(final_df['stopDateTime'])

final_df['duration'] = final_df['stopDateTime'] - final_df['startDateTime']
final_df['duration'] = final_df['duration'].apply(lambda x: x.total_seconds())

final_df['duration'] = final_df['duration'] / max(final_df['duration'])

final_df['destinationPort'] = final_df['destinationPort'] / max(final_df['destinationPort'])
final_df['sourcePort'] = final_df['sourcePort'] / max(final_df['sourcePort'])

final_df['totalDestinationBytes'] = final_df['totalDestinationBytes'] / max(final_df['totalDestinationBytes'])
final_df['totalDestinationPackets'] = final_df['totalDestinationPackets'] / max(final_df['totalDestinationPackets'])

final_df['totalSourceBytes'] = final_df['totalSourceBytes'] / max(final_df['totalSourceBytes'])
final_df['totalSourcePackets'] = final_df['totalSourcePackets'] / max(final_df['totalSourcePackets'])

final_df = final_df.drop(columns=['stopDateTime', 'startDateTime',
                                  'sourceTCPFlagsDescription',
                                  'destinationTCPFlagsDescription',
                                  'direction'])

final_df = final_df.sample(frac=1).reset_index(drop=True)

final_df.loc[final_df['appName'].
                 map(lambda x: x == 'PeerEnabler'), 'appName'] = 'P2P'

final_df.loc[final_df['appName'].
                 map(lambda x: x == 'BitTorrent'), 'appName'] = 'P2P'

final_df.loc[final_df['appName'].
                 map(lambda x: x == 'WindowsFileSharing'), 'appName'] = 'WinFileShare'

final_df.loc[final_df['appName'].
                 map(lambda x: x == 'WebMediaAudio'), 'appName'] = 'VoIP'

final_df.loc[final_df['appName'].
                 map(lambda x: x == 'WebMediaVideo'), 'appName'] = 'Streaming'

final_df.loc[final_df['appName'].
                 map(lambda x: x == 'WebMediaDocuments'), 'appName'] = 'Docs'

final_df.loc[final_df['appName'].
                 map(lambda x: x == 'MiscApplication'), 'appName'] = 'MiscApp'

final_df.loc[final_df['appName'].
                 map(lambda x: x == 'WebFileTransfer'), 'appName'] = 'FileTransfer'

temp = pd.DataFrame()
temp = temp.append(final_df[final_df.appName == 'NTP'])
temp = temp.append(final_df[final_df.appName == 'VoIP'])
temp = temp.append(final_df[final_df.appName == 'FileTransfer'])
temp = temp.append(final_df[final_df.appName == 'MiscApp'])
temp = temp.append(final_df[final_df.appName == 'Unknown_TCP'])
temp = temp.append(final_df[final_df.appName == 'Streaming'])
temp = temp.append(final_df[final_df.appName == 'P2P'])
temp = temp.append(final_df[final_df.appName == 'Docs'])
temp = temp.append(final_df[final_df.appName == 'ICMP'])
temp = temp.append(final_df[final_df.appName == 'SSH'])
temp = temp.append(final_df[final_df.appName == 'SMTP'])
temp = temp.append(final_df[final_df.appName == 'FTP'])
temp = temp.append(final_df[final_df.appName == 'POP'])
temp = temp.append(final_df[final_df.appName == 'Unknown_UDP'])
temp = temp.append(final_df[final_df.appName == 'IMAP'])
temp = temp.append(final_df[final_df.appName == 'WinFileShare'])
temp = temp.append(final_df[final_df.appName == 'DNS'])
temp = temp.append(final_df[final_df.appName == 'HTTPWeb'])
temp = temp.append(final_df[final_df.appName == 'HTTPImageTransfer'])

temp = temp.sample(frac=1).reset_index(drop=True)

train, test = train_test_split(temp, test_size=0.25)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

