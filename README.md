python -m pip install python-ev3dev2 //setup


//for at connecte bluetooth:
tænd bluetooth på din computer og ev3(den burde være tændt)
på ev3 remote network blutetooth og scan, find din computer og connect
tryk Windows + r og skriv npca.cpl og gå ind i bluetooth network connection. Højreklik på ev3dev og connect -> using accespoint
find din ipadresse i din kommandopromt ved at skrive ipconfig og find IPv4 ud fra bluetooth connection (måske ethernet)
i din kommandopromt skriv: ssh robot@<robot_ip> (indsæt fra kommandopromt) og koden: maker 
for at åbne en fil skriv: nano <navn_på_fil> og tryk ctrl + s for at gemme, så ctrl + x for at gå ud tryk y for at acceptere og derefter enter for at komme helt ud.
For at køre en program skriv: python3 <navn_på_fil> (det tager lidt tid at compile)