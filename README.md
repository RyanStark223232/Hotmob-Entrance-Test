# Optimize Ad Serving Strategy (Hotmob Entrance Test)

## How-to-Open

[Click here to view Analyst_Report.html](https://ryanstark223232.github.io/work-porfolio-hotmob/Entrance_Test/Analyst_Report.html)

## Background

This is the entrance test that I was required to complete prior to assuming my position as a Data Scientist at Hotmob. I was able to outperform other Data Scientists with over five years of experience.

The enclosed notebook serves as evidence of my capabilities in the following areas:

* Utilizing appropriate visualization tools to efficiently comprehend and preprocess data
* Employing tools such as correlation analysis to derive insights from data
* Effectively training machine learning models while performing hyperparameter tuning and model interpretation
* Converting model predictions into actionable insights for business problem-solving purposes.

## Test Content

```

### Objectives

1. Analyse the provided ad campaign for a food comapny and provide 1-3 data-driven actionable insights on improving camapign performance. 
You may also comment on data quality and make suggestions on ways for data enrichment/data acquisition that would help, or build a simple prediction model to predict potential campaign responders.

2. (Not Covered Here) The advertising industry rely heavily on specific pieces of strings (tracking browser cookies/mobile advertising ids) that are unique to each device or browser, which allow us to link different ad events to the same device/user and compose the respective digital footprint and user profiles. Amid the deprecation of tracking cookies and mobile advertising IDs, what other methods would you propose that can help us identify the same users from a stream of events without any identifier, given records of user location, url visited, app used, app installed, ip, user agent, and precise event timestamp? Briefly explain the methodolgy and potential challegnes that may arise.

### File Descriptions
- techtest_footprint.gz - csv of digital footprint of users 
- techtest_transactions.gz - csv of ad serving logs
- techtest_profiles.gz - csv of user profiles involved in the 2 campaigns
    - IAB segment source: https://www.google.com/search?q=iab+taxonomy+v2&oq=IAB+taxonomy&aqs=chrome.1.0l6j69i61l2.4588j0j7&sourceid=chrome&ie=UTF-8
- techtest_campaign_metadata.csv - campaign info. Contains ID used for mapping.
- Food Company - Creative Content Score.png - content scoring of Food Company Ad Content. This is based on ad image and text and gives you an idea of what the advertisement is about.

### Remarks
1. HMID is an ID that identify a specific device or a browser.
2. IAB Confidence: if cat is IAB, it means intensity of interests based on content analysis of their historical digital footprint, it starts from 0 but without upper limit. For 
3. netWorth: confidence range from 0 to 1, score below 0.6 is removed. NW0 meaning lowest and bottom 10% networth group. NW90 meaning top 10% group.
4. age and gender, the score is artificially assigned as 0.5.
5. listprice is the winning bidding price we paid (cost) for displaying the ad in a specific ad slot.
6. There could be missing user profiles. In addition, the profiles are created based on a longer time window that could fall outside of the given footprint data. They may or may not be useful for the analysis.
7. url provided are for your reference of users' digital footprint. Feel free to supplement with extrenal data.

```

## Results

I have exhibited essential skills that are necessary for a data scientist. Furthermore, I have accomplished the following:

* Streamlining the laborious process of model tuning through automation
* Communicating performance and decision-making through intuitive ROC curves and feature importance analysis
* Translating model predictions into actionable business insights for non-technical stakeholders

These achievements have placed me in a position of advantage over other professionals in this field.