I'm trying to create a simple "heatmap" style visualization app that can pull data from either the mega millions or powerball history and arrange the heatmap in the style of the lottery playing card. 
Please make it possible to choose between the games "Megamillions" vs "Powerball" As well as the state "layout"


Pictures of the various lotto card layouts are available in /ticketlayouts

The urls you can use for the underlying historical data is from the links here in csv format: 
url_mm_csv = "https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/megamillions.csv"
url_pb_csv = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/powerball.csv"

Local samples of that data is available at /data/texas

This doesnt have to be in python btw, thats just what I was using for my own development. 
In order to construct the heatmap, you should have 2 modes: 

1. is single date format to see the 6 numbers from that particular date overlaid on the card
2. is the date range format to see the counts of all numbers on the card over that selected date range

Otherwise - please maintain simplicity and ease of use / future feature development as the priority.