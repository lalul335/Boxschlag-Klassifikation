import matplotlib.pyplot as plt


def universal_plotter_for_single_axis(list_of_datasets,
                                      list_of_axis_to_plot,
                                      list_of_dataset_legend_titles,
                                      plot_title, y_axis_label, x_axis_label,
                                      figSizeTupel=(20, 10),
                                      saveImage=False,
                                      imageName='unknown_image.png'):
    """Plots a specific axis of multiple datasets (punches) of an given array. Converting mechanism is made for the notation style of the smartPunch project.

    Keyword arguments:
    list_of_datasets                -- List of datasets (type: list)
    list_of_axis_to_plot            -- List of strings containig the axis to plot for the dataset of specific array index (type: list)
    list_of_dataset_legend_titles   -- List strings containing the legend titles for each dataset axis (type: list)
    plot_title                      -- Plot title (type: string)
    y_axis_label                    -- Y-Axis label (type: string)
    x_axis_label                    -- X-Axis label (type: string)
    figSizeTupel                    -- Figure size as optional parameter (type: tupel, default: (20,10))
    saveImage                       -- If True the image is saved on the filesystem (type: boolean, default: False)
    imageName                       -- If saveImage is True, the image is saved using the given imageName (type: string, default: "unknown_image.png")

    Returns:
        void
    """

    fig, ax = plt.subplots(figsize=figSizeTupel)

    idx = 0
    for curDataSet in list_of_datasets:
        ax.plot(curDataSet['timestamp'].values, curDataSet[list_of_axis_to_plot[idx]],
                label=list_of_dataset_legend_titles[idx])
        idx += 1

    ax.set(xlabel=x_axis_label, ylabel=y_axis_label,
           title=plot_title)
    ax.grid()
    plt.legend()
    if saveImage:
        if imageName.endswith('.png'):
            plt.savefig(imageName)
        else:
            if "." not in imageName:
                name = imageName+'.png'
                plt.savefig(name)
            else:
                print("Error: File extension {} not allowed! See docs for more.".format(
                    imageName.split(".")[1]))
    plt.show()


def universal_plotter_for_all_axis(list_of_datasets,
                                   list_of_dataset_legend_titles,
                                   plot_title, y_axis_label, x_axis_label,
                                   figSizeTupel=(20, 10), saveImage=False,
                                   imageName='unknown_image.png'):
    """Plots all axis of multiple datasets (punches) of an given array. Converting mechanism is made for the notation style of the smartPunch project.

    Keyword arguments:
        list_of_datasets                -- List of datasets (type: list)
        list_of_dataset_legend_titles   -- List strings containing the legend titles for each dataset axis (type: list)
        plot_title                      -- Plot title (type: string)
        y_axis_label                    -- Y-Axis label (type: string)
        x_axis_label                    -- X-Axis label (type: string)
        figSizeTupel                    -- Figure size as optional parameter (type: tupel, default: (20,10))
        saveImage                       -- If True the image is saved on the filesystem (type: boolean, default: False)
        imageName                       -- If saveImage is True, the image is saved using the given imageName (type: string, default: "unknown_image.png")

    Returns:
        void
    """

    fig, ax = plt.subplots(figsize=figSizeTupel)

    idx = 0

    for curDataSet in list_of_datasets:
        universal_label = '%(titl)s (x-axis)' % {
            'titl': list_of_dataset_legend_titles[idx]} if list_of_dataset_legend_titles[idx] != '' else 'x-axis'
        ax.plot(curDataSet['timestamp'].values,
                curDataSet['x'], label=universal_label)
        universal_label = '%(titl)s (y-axis)' % {
            'titl': list_of_dataset_legend_titles[idx]} if list_of_dataset_legend_titles[idx] != '' else 'y-axis'
        ax.plot(curDataSet['timestamp'].values,
                curDataSet['y'], label=universal_label)
        universal_label = '%(titl)s (z-axis)' % {
            'titl': list_of_dataset_legend_titles[idx]} if list_of_dataset_legend_titles[idx] != '' else 'z-axis'
        ax.plot(curDataSet['timestamp'].values,
                curDataSet['z'], label=universal_label)
        idx += 1

    ax.set(xlabel=x_axis_label, ylabel=y_axis_label,
           title=plot_title)
    ax.grid()
    plt.legend()
    if saveImage:
        if imageName.endswith('.png'):
            plt.savefig(imageName)
        else:
            if "." not in imageName:
                name = imageName+'.png'
                plt.savefig(name)
            else:
                print("Error: File extension {} not allowed! See docs for more.".format(
                    imageName.split(".")[1]))
    plt.show()


def single_plot_single_axis(dataset, axis, plot_title, y_axis_label, x_axis_label, default_axis_legend=True, figSizeTupel=(20, 10), saveImage=False,
                            imageName='unknown_image.png'):
    """Plots a single axis of a single dataset (punch). Converting mechanism is made for the notation style of the smartPunch project.

    Keyword arguments:
        dataset                         -- Dataset (single punch) to plot (type: pandas.Dataframe)
        axis                            -- Axis to plot (type: string)
        default_axis_legend             -- If not False this parameter is used for the axis legend title instead of the default value (type: string, default: True)
        plot_title                      -- Plot title (type: string)
        y_axis_label                    -- Y-Axis label (type: string)
        x_axis_label                    -- X-Axis label (type: string)
        figSizeTupel                    -- Figure size as optional parameter (type: tupel, default: (20,10))
        saveImage                       -- If True the image is saved on the filesystem (type: boolean, default: False)
        imageName                       -- If saveImage is True, the image is saved using the given imageName (type: string, default: "unknown_image.png")

    Returns:
        void
    """
    fig, ax = plt.subplots(figsize=figSizeTupel)

    universal_label = ' %(ax)s-axis' % {
        'ax': axis} if default_axis_legend == True else default_axis_legend

    ax.plot(dataset['timestamp'].values, dataset[axis], label=universal_label)
    ax.set(xlabel=x_axis_label, ylabel=y_axis_label,
           title=plot_title)
    ax.grid()
    plt.legend()
    if saveImage:
        if imageName.endswith('.png'):
            plt.savefig(imageName)
        else:
            if "." not in imageName:
                name = imageName+'.png'
                plt.savefig(name)
            else:
                print("Error: File extension {} not allowed! See docs for more.".format(
                    imageName.split(".")[1]))
    plt.show()


def single_plot_all_axis(dataset, plot_title, y_axis_label, x_axis_label, default_axis_legend=True, figSizeTupel=(20, 10), saveImage=False,
                         imageName='unknown_image.png'):
    """Plots all axis of a single dataset (punch). Converting mechanism is made for the notation style of the smartPunch project.

    Keyword arguments:
        dataset                         -- Dataset (punch) to plot (type: pandas.Dataframe)
        default_axis_legend             -- If not False this parameter is used for the axis legend title instead of the default value (type: string, default: True)
        plot_title                      -- Plot title (type: string)
        y_axis_label                    -- Y-Axis label (type: string)
        x_axis_label                    -- X-Axis label (type: string)
        figSizeTupel                    -- Figure size as optional parameter (type: tupel, default: (20,10))
        saveImage                       -- If True the image is saved on the filesystem (type: boolean, default: False)
        imageName                       -- If saveImage is True, the image is saved using the given imageName (type: string, default: "unknown_image.png")

    Returns:
        void
    """
    fig, ax = plt.subplots(figsize=figSizeTupel)

    universal_label = 'x-axis' if default_axis_legend == True else default_axis_legend[0]
    ax.plot(dataset['timestamp'].values, dataset['x'], label=universal_label)
    universal_label = 'y-axis' if default_axis_legend == True else default_axis_legend[1]
    ax.plot(dataset['timestamp'].values, dataset['y'], label=universal_label)
    universal_label = 'z-axis' if default_axis_legend == True else default_axis_legend[2]
    ax.plot(dataset['timestamp'].values, dataset['z'], label=universal_label)

    ax.set(xlabel=x_axis_label, ylabel=y_axis_label,
           title=plot_title)
    ax.grid()
    plt.legend()
    if saveImage:
        if imageName.endswith('.png'):
            plt.savefig(imageName)
        else:
            if "." not in imageName:
                name = imageName+'.png'
                plt.savefig(name)
            else:
                print("Error: File extension {} not allowed! See docs for more.".format(
                    imageName.split(".")[1]))
    plt.show()


def print_usage_info():
    print("use the functions in the notation of the following examples:")
    print(
        "universal_plotter_for_single_axis([ds[0],...,ds[n]],['axisToPrintForDs1','axisToPrintForDsN'],['ds[0] legend text','ds[N] legend text'],'Plot-Title','y-Axis description','y-Axis description')")
    print(
        "universal_plotter_for_all_axis([ds[0],...,ds[n]],['ds[0] legend text','ds[N] legend text'],'Plot-Title','y-Axis description','x-Axis description')")
    print("single_plot_single_axis(dataset,'axisToPrint','Plot-Title','y-Axis description','x-Axis description','default_axis_legend='legend text')")
    print("single_plot_all_axis(dataset,'Plot-Title','y-Axis description','x-Axis description','default_axis_legend='legend text')")
