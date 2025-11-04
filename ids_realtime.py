import pyshark

capture = pyshark.LiveCapture(interface='Wi-Fi') #Ethernet
for packet in capture.sniff_continuously(packet_count=5):
    print(packet.highest_layer, packet.length, packet.sniff_time)