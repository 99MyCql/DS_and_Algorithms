#include<stdio.h>
#include<stdlib.h>

typedef struct LNode{
	int data;
	LNode *next;
}*LinkList, LNode;

//在pos位置插入e 
void insertLinkList(LinkList &list, int pos, LNode *e){
	int i = 1;
	LNode *p = list->next, *prior = list;
	while(p && i<pos){
		prior = p;
		p = prior->next;
		i++;
	}
	prior->next = e;
	e->next = p;
	list->data++;
}

int main(){
	
	int n = 0, m = 0;
	scanf("%d%d", &n, &m);
	
	//初始化 
	LinkList list = (LNode*)malloc(sizeof(LNode));
	list->data = 0;
	LNode *temp, *p = list;
	for(int i=0; i<n; i++){
		temp = (LNode*)malloc(sizeof(LNode));
		temp->data = i+1;					//将学号赋给结点的数据域 
		p->next = temp;
		p = p->next;
		list->data++;
	}
	p->next = NULL;
	
	//开始排队 
	int num = 0, pos = 0;
	int j = 0;
	LNode *prior = NULL;
	for(int i=1; i<=m; i++){
		scanf("%d%d", &num, &pos);
		p = list->next;
		prior = list;
		j = 1;
		while(p){
			if(p->data == num){				//找到等于该学号的结点 
				prior->next = p->next;		//将该结点取出 
				list->data--;
				insertLinkList(list, j+pos, p);
			}
			prior = p;
			p = prior->next;
			j++;
		} 
	}
	p = list->next;
	int i = 1;
	while(p) {
		printf("%d ", p->data); 
		p = p->next;
		i++;
	}
	
	//销毁链表 
	p = list;
	while(p) {
		temp = p;
		p = p->next;
		free(temp);
	}
}
