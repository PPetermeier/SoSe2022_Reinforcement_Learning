# Utilities
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Classes needed to run experiment
from Assignment3 import RLrunner, Random_agent

def run_training()-> None:
    """Bundles both training runs into one function.
    Training progress will be logged by tensorflow itself and can be accessed via tensorboard"""
    runner = RLrunner()
    runner.run_cartpole()
    runner.run_dueling_cartpole()

def run_tests(number)-> None:
    """Bundles both tests into one function. Number is given to set amount of tests to run."""
    runner = RLrunner()
    runner.test(number)
    runner.test_dueling(number)

def visualize(episodecount)-> None:
    """Takes recorded datasets, aggregates them as needed and creates the plots for presentation
    Also runs the random agent the number of times given, which has to match with the number of episodes
    in the data for the visualization to make sense.
    All plots are already saved as .png, this is mainly for reproducibility.
    Also note that this code just creates the plot from our main, successful run, not run1 and the 40 episodes one"""
    # Run random_agent to create data as needed
    random_agent = Random_agent(test_episodes=episodecount)
    random_test = random_agent.test()
    # Aggregate Data from records, combine with random data created, give proper names and order as wanted
    raw_data = pd.read_json('./jsons/vanilla_run4_reward.json')
    dueling = pd.read_json('./jsons/dueling_run4_reward.json')
    raw_data = raw_data.rename(columns={1: 'Episode', 2: 'Vanilla'})
    raw_data['Dueling'] = dueling[2]
    raw_data['Random'] = random_test
    raw_data = raw_data[['Episode', 'Random', 'Vanilla', 'Dueling']]
    print(raw_data.describe())
    # Create the trend data from the raw data, take note that order is == as in raw_data
    trends = pd.DataFrame()
    trends['Episode'] = raw_data['Episode']
    trends['Random Trend'] = raw_data['Random'].rolling(10).sum()/10
    trends['Vanilla Trend'] = raw_data['Vanilla'].rolling(10).sum()/10
    trends['Dueling Trend'] = raw_data['Dueling'].rolling(10).sum()/10
    trends = trends.fillna(0)
    # Create advantage dataframe by subtracting vanilla-values from dueling values
    advantage = pd.DataFrame()
    advantage['Episode'] = raw_data['Episode']
    advantage['Raw'] = raw_data['Dueling']-raw_data['Vanilla']
    advantage = advantage.fillna(0)
    advantage_trend = pd.DataFrame()
    advantage_trend['Episode'] = raw_data['Episode']
    advantage_trend['Advantage-Window of Dueling in Trend Window 10'] = trends['Dueling Trend']-trends['Vanilla Trend']
    advantage_trend = advantage_trend.fillna(0)
    # 'Melt' Dataframe in order to be able to use type of agent as hue
    df_melted = raw_data.melt('Episode', var_name='Type', value_name='Reward')
    trends_melted = trends.melt('Episode', var_name='Type', value_name='Reward')
    trends_melted = trends_melted.fillna(0)
    # Configure seaborn the way we want it, plots are created in the same order as used in presentation
    palette = sns.color_palette('Set2')
    sns.set_palette('Set2')
    # -------------
    fig = plt.gcf()
    fig.set_size_inches(30, 6)
    plot1 = sns.lineplot(data=df_melted, x='Episode', y='Reward', hue='Type')
    plt.show()
    plt.clf()
    # -------------
    fig3 = plt.gcf()
    fig3.set_size_inches(20, 6)
    stripplot = sns.stripplot(data=df_melted, x='Reward', y='Type', size=3)
    plt.show()
    plt.clf()
    # -------------
    fig4 = plt.gcf()
    fig4.set_size_inches(20, 6)
    boxplot = sns.boxplot(data=df_melted, x='Reward', y='Type', )
    plt.show()
    plt.clf()
    # -------------
    fig2 = plt.gcf()
    fig2.set_size_inches(30, 6)
    plot2 = sns.lineplot(data=trends_melted, x='Episode', y='Reward', hue='Type',)
    plot2.set(ylabel='Average reward in a 10-Episode Window')
    plt.show()
    plt.clf()
    # -------------
    fig5 = plt.gcf()
    fig5.set_size_inches(30, 6)
    plot4 = sns.lineplot(data=advantage, x='Episode', y='Raw', color=palette[2])
    plot4.fill_between(advantage['Episode'], advantage['Raw'], color=palette[2])
    plot4.set(ylabel='Reward Difference between Dueling & Vanilla')
    plt.show()
    # -------------
if __name__ == '__main__':
    run_tests(1)
