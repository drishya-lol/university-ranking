import matplotlib.pyplot as plt
import pandas as pd

universityList = pd.read_csv('cwurData.csv')

for i in range(2012, 2016):
    uniYear = universityList[universityList['year'] == i]

    topUniversities = uniYear.iloc[:20, :]  # Now selecting top 20 universities
    print(topUniversities)

    xvals = topUniversities['world_rank']
    yvals_alum = topUniversities['alumni_employment']
    yvals_citations = topUniversities['citations']
    yvals_patents = topUniversities['patents']
    yvals_quality = topUniversities['quality_of_education']  # Example of a new column
    university_names = topUniversities['institution'].values

    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'pink', 'orange', 'purple', 'gray', 'lime', 'maroon', 'navy', 'olive', 'teal', 'aqua', 'fuchsia', 'silver', 'gold']

    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 8))  # Adjusted for 4 subplots

    # Plot for Alumni Employment
    axs[0].bar(xvals, yvals_alum, color=colors[:20])
    axs[0].set_title(f'Alumni Employment {i}')
    axs[0].set_xlabel('World Rank')
    axs[0].set_ylabel('Alumni Employment')
    axs[0].set_xticks(xvals)
    axs[0].set_xticklabels(university_names, rotation=90, ha='right')

    # Plot for Citations
    axs[1].bar(xvals, yvals_citations, color=colors[:20])
    axs[1].set_title(f'Citations {i}')
    axs[1].set_xlabel('World Rank')
    axs[1].set_ylabel('Citations')
    axs[1].set_xticks(xvals)
    axs[1].set_xticklabels(university_names, rotation=90, ha='right')

    # Plot for Patents
    axs[2].bar(xvals, yvals_patents, color=colors[:20])
    axs[2].set_title(f'Patents {i}')
    axs[2].set_xlabel('World Rank')
    axs[2].set_ylabel('Patents')
    axs[2].set_xticks(xvals)
    axs[2].set_xticklabels(university_names, rotation=90, ha='right')

    # Plot for Quality of Education
    axs[3].bar(xvals, yvals_quality, color=colors[:20])
    axs[3].set_title(f'Quality of Education {i}')
    axs[3].set_xlabel('World Rank')
    axs[3].set_ylabel('Quality of Education')
    axs[3].set_xticks(xvals)
    axs[3].set_xticklabels(university_names, rotation=90, ha='right')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    plt.show()