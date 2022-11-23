# TalentSquad-III

## Background

A study has been carried out to see if the academic performance of children is influenced by the academic level of their parents. Therefore, the academic results of the students will be evaluated based on several variables

## Problem

This is a multiclass classification problem in which the biggest challenge is to predict the educational level of the parents without knowing if it is directly correlated with the variables that we are analyzing.

## Results

We found that the grades were somewhat related to the educational level of the parents, but this was not a determining factor. In turn, according to the data, the socioeconomic level obtained from access to scholarships or private classes does not seem to be influenced by the educational level of the parents. Although it does correlate with the grades obtained, this factor does not help us to determine the educational level of the parents based on the attributes.

## Analysis

First, a correlation analysis of variables was carried out and a Lasso model was implemented, which was not able to determine which variables had the most influence, so different systems were tested, resulting in better results with Gaussian Naive Bayes, even though these were really bad. Well, as we have said, the variables are not related to the class.

## Solution

Based on the provided data, the performance of children is NOT influenced by the academic level of their parents.
A Gaussian Naive Bayes classifier with additional columns based on the provided data.

### License
Public.
