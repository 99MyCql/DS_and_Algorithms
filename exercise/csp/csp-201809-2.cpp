#include<stdio.h>
int main()
{
	int h[4000];
	int w[4000];
	int n=0;
	int res=0;
	scanf("%d",&n);
	for(int i=0;i<(2*n);i++) scanf("%d",&h[i]);
	for(int i=0;i<(2*n);i++) scanf("%d",&w[i]);
	for(int i=0;i<(2*n);i+=2)
	{
		for(int j=0;j<(2*n);j+=2)
		{
			if(h[i+1]<=w[j]) break;
			if(h[i]>=w[j+1]) continue;
			if(h[i]<w[j+1]&&h[i]>=w[j])
			{
				if(h[i+1]>=w[j+1]) res+=(w[j+1]-h[i]);
				else res+=(h[i+1]-h[i]);
				continue;
			}
			if(h[i+1]>w[j]&&h[i]<=w[j])
			{
				if(w[j+1]>h[i+1]) res+=(h[i+1]-w[j]);
				else res+=(w[j+1]-w[j]);
				continue;
			}
		}
	}
	printf("%d",res);
	return 0;
}
