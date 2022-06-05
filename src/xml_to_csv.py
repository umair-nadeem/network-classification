import os
import itertools
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.style.use('seaborn')
os.chdir('/home/umair/PycharmProjects/thesis/files/classification_data/xmls')

appName = np.asarray([])
totalSourceBytes = np.asarray([])
totalDestinationBytes = np.asarray([])
totalDestinationPackets = np.asarray([])
totalSourcePackets = np.asarray([])
direction = np.asarray([])
sourceTCPFlagsDescription = np.asarray([])
destinationTCPFlagsDescription = np.asarray([])
source = np.asarray([])
protocolName = np.asarray([])
sourcePort = np.asarray([])
destination = np.asarray([])
destinationPort = np.asarray([])
startDateTime = np.asarray([])
stopDateTime = np.asarray([])
Tag = np.asarray([])

XML = 'TestbedSunJun13Flows.xml'
element_name = XML[:-4]

tree = ET.parse(XML)
root = tree.getroot()

elements = root.findall(element_name)

print(len(elements))

for i, element in enumerate(elements):
    if (element.findtext('appName') == 'WebMediaVideo') | (element.findtext('appName') == 'WebMediaAudio'):
        appName = np.append(appName, element.findtext('appName'))
        totalSourceBytes = np.append(totalSourceBytes, element.findtext('totalSourceBytes'))
        totalDestinationBytes = np.append(totalDestinationBytes, element.findtext('totalDestinationBytes'))
        totalDestinationPackets = np.append(totalDestinationPackets, element.findtext('totalDestinationPackets'))
        totalSourcePackets = np.append(totalSourcePackets, element.findtext('totalSourcePackets'))
        direction = np.append(direction, element.findtext('direction'))
        sourceTCPFlagsDescription = np.append(sourceTCPFlagsDescription,
                                              element.findtext('sourceTCPFlagsDescription'))
        destinationTCPFlagsDescription = np.append(destinationTCPFlagsDescription,
                                                   element.findtext('destinationTCPFlagsDescription'))
        source = np.append(source, element.findtext('source'))
        sourcePort = np.append(sourcePort, element.findtext('sourcePort'))
        destination = np.append(destination, element.findtext('destination'))
        destinationPort = np.append(destinationPort, element.findtext('destinationPort'))
        protocolName = np.append(protocolName, element.findtext('protocolName'))
        startDateTime = np.append(startDateTime, element.findtext('startDateTime'))
        stopDateTime = np.append(stopDateTime, element.findtext('stopDateTime'))
        Tag = np.append(Tag, element.findtext('Tag'))

df = pd.DataFrame({'appName': appName,
                   'totalSourceBytes': totalSourceBytes,
                   'totalDestinationBytes': totalDestinationBytes,
                   'totalDestinationPackets': totalDestinationPackets,
                   'totalSourcePackets': totalSourcePackets,
                   'direction': direction,
                   'sourceTCPFlagsDescription': sourceTCPFlagsDescription,
                   'destinationTCPFlagsDescription': destinationTCPFlagsDescription,
                   'source': source,
                   'sourcePort': sourcePort,
                   'destination': destination,
                   'destinationPort': destinationPort,
                   'protocolName': protocolName,
                   'startDateTime': startDateTime,
                   'stopDateTime': stopDateTime,
                   'Tag': Tag,
                   })

csv = XML[:-3] + 'csv'
df.to_csv(csv, index=None)

# for audio: sun: 4, thu-2: 28, tue-1: 8, tue-2: 2, tue-3: 0

# for video: sun: 0, thu-2: 235, tue-1: 0, tue-2: 3, tue-3: 2