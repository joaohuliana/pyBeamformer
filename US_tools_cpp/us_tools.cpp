#include <cmath>

extern "C" {
	
	/*int signum(double x) {
		if (x > 0) return 1;
		if (x < 0) return -1;
		return 0;
		}*/
	int signum(double x) {
		return (x > 0) - (x < 0);
	}
	
	void delaymap(double delays[], const double prm[]) {
		int Z = prm[0];
		int X = prm[1];
		double dx = prm[2]; //transducer's element pitch[mm]
		double c = prm[3]; //speed of sound [m/s]
		double sf = prm[4]; //sampling frequency [Hz]		
		double dz = (c*(1/sf)*1000)/2;
		
		double z_pos;
		double x_pos;
		for(int z = 0; z < Z; z++) {
			z_pos = z*dz;
			for(int x = 0; x < 2*X; x++) {
				x_pos = (X-x)*dx;
				delays[z*(2*X)+x]=(int)((sqrt(z_pos*z_pos + x_pos*x_pos) - z_pos)/(2*dz));
			}
		}
	}
	void delaymapPA(double delays[], const double prm[]) {
		int Z = prm[0];
		int X = prm[1];
		double dx = prm[2]; //transducer's element pitch[mm]
		double c = prm[3]; //speed of sound [m/s]
		double sf = prm[4]; //sampling frequency [Hz]		
		double dz = (c*(1/sf)*1000);
		
		double z_pos;
		double x_pos;
		for(int z = 0; z < Z; z++) {
			z_pos = z*dz;
			for(int x = 0; x < 2*X; x++) {
				x_pos = (X-x)*dx;
				delays[z*(2*X)+x]=(int)((sqrt(z_pos*z_pos + x_pos*x_pos) - z_pos)/(dz));
			}
		}
	}
    
	void das(double bfData[], const int rawData[], const int delays[], const double prm[]) {
        int Z = prm[0];
		int X = prm[1];
		double dz = prm[2];
		double dx = prm[3];
		int aperture = prm[4];
		double ctag = prm[5];	
		
		double acut = (dz / dx) * tan((ctag * M_PI) / 180.0);

		// Calculate zlim
		int zlim = static_cast<int>(aperture / acut);
		
		double count;
		int realap;
		int i_d = 0;
		int z = 0;
		int m = 0;
		int hap = (int)(aperture/2); 
		while(z < Z) {
			if(z < zlim)
				realap = (int)(((acut/2)*z)+1);
			else
				realap = hap;
			for(int x = 0; x < X; x++){
				count = 0;
				for (int i = (x - realap); i <= (x + realap); i++){
					if(i >= 0 && i < X){
						i_d = z + delays[z*(2*X)+(((X)-x)+i)];
						if(i_d < Z)
							bfData[m*X+x] = bfData[m*X+x] + rawData[i_d*X+i];
						else
							bfData[m*X+x] = bfData[m*X+x] + 0;
						count = count + 1;
					}				
				}
				bfData[m*X+x] = bfData[m*X+x]/(count);
		    }
			m = m+1;
			z = z+1;
		}
		
    }
	
	void dmas(double bfData[], const int rawData[], const int delays[], const double prm[]) {
        int Z = prm[0];
		int X = prm[1];
		double dz = prm[2];
		double dx = prm[3];
		int aperture = prm[4];
		double ctag = prm[5];	
		
		double acut = (dz / dx) * tan((ctag * M_PI) / 180.0);

		// Calculate zlim
		int zlim = static_cast<int>(aperture / acut);
		
		double count;
		int realap;
		int i_d = 0;
		int i_dj = 0;
		int z = 0;
		int m = 0;
		int hap = (int)(aperture/2);
		while(z < Z) {
			if(z < zlim)
				realap = (int)(((acut/2)*(z))+1);
			else
				realap = hap;
			for(int x = 0; x < X; x++) {
				double count = 1;
				for (int i = (x - realap); i <= (x + realap); i++){
					if(i >= 0 && i < (X-1)){
						i_d = z + delays[z*(2*X)+(((X)-x)+i)];
						for (int j = (i+1); j <= (x + realap); j++){
							if(j < X){
								i_dj = z + (int)delays[z*(2*X)+(((X)-x)+j)];
								if(i_d < Z && i_dj <Z)
									bfData[m*X+x] = bfData[m*X+x] + rawData[i_d*X+i]*rawData[i_dj*X+j];
								else
									bfData[m*X+x] = bfData[m*X+x] + 0;								
								count = count +1;							
							}							
						}
					}
				}
				bfData[m*X+x] = bfData[m*X+x]/(count);
			}	
			m = m+1;
			z = z+1;
		}
		
    }
	void fdmas(double bfData[], const int rawData[], const int delays[], const double prm[]) {
        int Z = prm[0];
		int X = prm[1];
		double dz = prm[2];
		double dx = prm[3];
		int aperture = prm[4];
		double ctag = prm[5];	
		
		double acut = (dz / dx) * tan((ctag * M_PI) / 180.0);

		// Calculate zlim
		int zlim = static_cast<int>(aperture / acut);
		
		double count;
		int realap;
		int i_d = 0;
		int i_dj = 0;
		int z = 0;
		int m = 0;
		int hap = (int)(aperture/2);
		while(z < Z) {
			if(z < zlim)
				realap = (int)(((acut/2)*(z))+1);
			else
				realap = hap;
			for(int x = 0; x < X; x++) {
				double count = 1;
				for (int i = (x - realap); i <= (x + realap); i++){
					if(i >= 0 && i < (X-1)){
						i_d = z + delays[z*(2*X)+(((X)-x)+i)];
						for (int j = (i+1); j <= (x + realap); j++){
							if(j < X){
								i_dj = z + (int)delays[z*(2*X)+(((X)-x)+j)];
								if(i_d < Z && i_dj <Z)
									//bfData[m*X+x] = bfData[m*X+x] + rawData[i_d*X+i]*rawData[i_dj*X+j];
									bfData[m*X+x] = bfData[m*X+x] + signum(rawData[i_d*X+i]*rawData[i_dj*X+j])*sqrt(abs(rawData[i_d*X+i]*rawData[i_dj*X+j]));
								else
									bfData[m*X+x] = bfData[m*X+x] + 0;								
								count = count +1;							
							}							
						}
					}
				}
				bfData[m*X+x] = bfData[m*X+x]/(count);
			}	
			m = m+1;
			z = z+1;
		}
		
    }
	
}
