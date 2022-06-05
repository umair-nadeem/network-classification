#!/bin/bash
for i in $(ls -d /home/umair/PycharmProjects/files/dumps/);
do
	date
	echo item: $i
	#editcap -i 3600 -F pcap $i $i.pcap
	#tshark -r $i -T fields -E quote=d -e frame.time -e frame.len > $i.txt
	#tshark -r $i -T fields -E separator=, -e frame.time -e frame.len -e frame.protocols -e ip.version -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport > $i.txt

done
