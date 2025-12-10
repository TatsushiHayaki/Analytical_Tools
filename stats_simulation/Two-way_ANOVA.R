#https://www.youtube.com/watch?v=ajLdnsLPErE
#two-way anova
#what determines the variations of test scores,
#age group, gender, or both?
data = data.frame(
  gender = c("B","B","B","G","G","G",
             "B","B","B","G","G","G",
             "B","B","B","G","G","G"),
  score = c(4,6,8,4,8,9,6,6,9,7,10,13,8,9,13,12,14,16),
  group =c("10ysd","10ysd","10ysd","10ysd","10ysd","10ysd",
           "11ysd","11ysd","11ysd","11ysd","11ysd","11ysd",
           "12ysd","12ysd","12ysd","12ysd","12ysd","12ysd"))
lm =lm (score ~ gender + group + gender*group)
anova(lm)

data$score[data$gender =="B"] #this way of coding is tedious
attach(data)

mu_B = mean(score[gender =="B"])
mu_G = mean(score[gender =="G"])

mu_10 = mean(score[group =="10ysd"])
mu_11 = mean(score[group =="11ysd"])
mu_12 = mean(score[group =="12ysd"])
mu = mean(score)

mu_10_B = mean(score[group =="10ysd" & gender == "B"])
mu_10_G =mean(score[group =="10ysd" & gender == "G"])
mu_11_B =mean(score[group =="11ysd" & gender == "B"])
mu_11_G =mean(score[group =="11ysd" & gender == "G"])
mu_12_B =mean(score[group =="12ysd" & gender == "B"])
mu_12_G =mean(score[group =="12ysd" & gender == "G"])

meantable = rbind(c(mu_10_B,mu_11_B,mu_12_B,mu_B),
        c(mu_10_G,mu_11_G,mu_12_G,mu_G),
        c(mu_10,mu_11,mu_12,mu))
dimnames(meantable) = list(c("B","G","Average"),
                           c("10ysd","11ysd","12ysd","Average"))

#sum of squares 1st factor (Gender)
SS_gender =
  9 * (mu_B - mu)^2  + 
  9 * (mu_G - mu)^2 #9 is the number of Boys and Girls, resp. 
df_gender = 2-1 # (Boys and Girls) -1

#sum of squares 2st factor (Age)
SS_age =
  6 * (mu_10 - mu)^2 + 
  6 * (mu_11 - mu)^2 + 
  6 * (mu_12 - mu)^2   #6 is the number of obs in each Age group.
df_age = 3-1 #(three types of group) -1

#sum of squares Both factors (Age & Gender) 
SS_age_gender = 3* (
sum(mu_10_B - mu_10 - mu_B + mu)^2 + 
sum(mu_11_B - mu_11 - mu_B + mu)^2 +
sum(mu_12_B - mu_12 - mu_B + mu)^2 +
sum(mu_10_G - mu_10 - mu_G + mu)^2 +
sum(mu_11_G - mu_11 - mu_G + mu)^2 +
sum(mu_12_G - mu_12 - mu_G + mu)^2)
df_age_gender = df_gender * df_age

#sum of squares within (Error)
SS_within_error = 
sum((score[gender == "B" & group == "10ysd"] - mu_10_B)^2) + 
sum((score[gender == "B" & group == "11ysd"] - mu_11_B)^2) + 
sum((score[gender == "B" & group == "12ysd"] - mu_12_B)^2) +
sum((score[gender == "G" & group == "10ysd"] - mu_10_G)^2) + 
sum((score[gender == "G" & group == "11ysd"] - mu_11_G)^2) + 
sum((score[gender == "G" & group == "12ysd"] - mu_12_G)^2)  
df_within_error = 6*(3 - 1) # in each 6 category there are three observations.

#sum of squares in total
SS_total = sum((score - mu)^2) 
df_total = nrow(data) -1

#check
SS_total - (SS_age + SS_gender + SS_age_gender + SS_within_error)

#F ratio
F_gender = (SS_gender/df_gender)/(SS_within_error/df_within_error) 
qf(0.95, df1 = df_gender, df2 = df_within_error) #critical value
F_age = (SS_age/df_age)/(SS_within_error/df_within_error)
qf(0.95, df1 = df_age, df2 = df_within_error) #critical value
F_age_gender = (SS_age_gender/df_age_gender)/(SS_within_error/df_within_error) 
qf(0.95, df1 = df_age_gender, df2 = df_within_error) #Fail to reject the H that "Gender and Age interaction will have no significant effect on scores. 
"

