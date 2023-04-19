from datetime import datetime

def relative_time():
   for trip in triplist:
      gmttime = globals()['Trip' + '_' + str(trip[0]) + '_' + str(trip[1])]['GMT time'].tolist()
      Times = []
      for i in range(0, len(gmttime)):
         t = (datetime.strptime(str(int(gmttime[i])).zfill(6), "%H%M%S") - datetime.strptime(
            str(int(gmttime[0])).zfill(6), "%H%M%S")).total_seconds()
         Times.append(t)
      globals()['Trip' + '_' + str(trip[0]) + '_' + str(trip[1])]['relative time'] = Times