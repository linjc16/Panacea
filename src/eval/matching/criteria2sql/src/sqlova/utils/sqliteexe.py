#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:39:05 2019

@author: yu
"""

#generate table.jsonl
import numpy as np
import pandas as pd
import csv
import json
import nltk
import json_lines
import copy
import sqlite3
import os



def QueryFormat(sql_format1,hds):
    sql=[]
    sql_format=copy.deepcopy(sql_format1)
    sql.append(sql_format['conds'][0])
    for i in range(1,len(sql_format['conds'])):
        sql.append([sql_format['operator'][i],sql_format['priority'][i]])
        sql.append(sql_format['conds'][i])

    column_name=copy.deepcopy(hds)
    for i in range(0, len(column_name)):
        column_name[i]=column_name[i].replace(' ', '_')

    cond_ops = ['=', '>', '<', '>=', '<=', '<>', '']
    comb_ops = ['and','or','+', '>', '>=','']
    part_ops = ['(', ')']

    #start parse on tokens
    bucket=[] #temporary store emelemnts in single condition
    condition=[] #temporary store emelemnts in single condition

    pre_item=''
#recover index to text
    for i in range(0,len(sql)):
        #map index to text in operator
        if len(sql[i])==2:
            sql[i][0]=comb_ops[sql[i][0]]

        if len(sql[i])==3:
            if sql[i][1]== 6:
                sql[i][1]=''
                sql[i][0]=''
            else:
                sql[i][1]=cond_ops[sql[i][1]]
                sql[i][0]=column_name[sql[i][0]]


#now convertion
    for i in range(0, len(sql)):
        if len(sql[i])==3:
            sql[i]=str(sql[i][0])+' '+str(sql[i][1])+' '+str(sql[i][2])
        if len(sql[i])==1:
            sql[i]=str(sql[i][0])
#encode with index
    text=[]
#find the max priority
    max_prior=0
    for item in sql:
        if len(item)==2:
            if max_prior<item[1]:
                max_prior=item[1]
#find max priority
    for prior in range(max_prior,-1,-1):
        ind=[]
#find the index of operator with same priority
        for i in range(0,len(sql)):
            if len(sql[i])==2:
                if sql[i][1]==prior:
                    ind.append(i)

        if prior ==0:
            for j in range(0,len(ind)):
                sql[ind[j]]=str(sql[ind[j]][0])
            break
#record the position of perenthesis for that priority
        part_l=[]
        part_r=[]
        if ind != []:
            left=0
            pre=left

            part_l.append(ind[left]-1)


            if len(ind)==1:
                 part_r.append(ind[pre]+1)

            for i in range(left+1, len(ind)):
                if ind[i]-ind[pre]==2 and sql[ind[i]][0]:
                    pre=i
                else :
                    part_r.append(ind[pre]+1)
                    left=i
                    pre=left
                    part_l.append(ind[left]-1)
                if i == len(ind)-1:
                    part_r.append(ind[pre]+1)

#insert parenthesis into the sql and merge tokens
        for i in range(0,len(part_l)):
            l=part_l[i]
            r=part_r[i]
            for j in range(l,r+1):
                if len(sql[j])==2:
                    sql[j]=str(sql[j][0])

            group=' '.join(sql[l:r+1])
            if sql[l+1]=='+':
               group='case when '+ group + ' then 1 else 0 end'
               group=group.replace('+','then 1 else 0 end + case when')
            group=' ( '+group+' ) '

            for j in range(l,r+1):
                sql[j]=' '
            sql[l]=group
        while ' ' in sql:
            sql.remove(' ')
# join tokens into string
    text=' '.join(sql)
    tokens=text.split(' ')
    sql=' '.join(tokens)

# remove redundant parenthesis
    tokens=[]
    tokens=sql.split(' ')
    while ' ' in tokens:
        tokens.remove(' ')
    while '' in tokens:
        tokens.remove('')
    when_list=[]
    then_list=[]
    for i in range(0,len(tokens)):
        if tokens[i] == 'when':
            when_list.append(i)

        if tokens[i] == 'then':
            then_list.append(i)

    for i in range(0,len(when_list)):
        l=when_list[i]
        r=then_list[i]
        for j in range(1,r-l):
            if tokens[l+j]=='(' and tokens[r-j]==')':
                cnt=0
                flg=0
                for idx in range(l+j,r-j):
                    if tokens[idx]=='(':
                        cnt=cnt+1
                    elif tokens[idx]==')':
                        cnt=cnt-1
                    if cnt==0:
                        flg=1
                        break
                if flg==0:
                    tokens[l+j]=' '
                    tokens[r-j]=' '
            else:
                break

    while ' ' in tokens:
        tokens.remove(' ')
    while '' in tokens:
        tokens.remove('')

    sql=' '.join(tokens)

    return sql



def execute(sqls,hds):
    # db_path=os.path.abspath()
    conn = sqlite3.connect('../data/records.db')
    c = conn.cursor()
    query=QueryFormat(sqls,hds)

    cursor = c.execute( 'SELECT id from records where ('+query+')' )
    res=[]
    for row in cursor:
        res.append(row[0])
    return res
