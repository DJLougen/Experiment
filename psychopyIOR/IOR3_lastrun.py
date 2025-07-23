#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Wed Jul 23 10:11:59 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'IOR _HSF5'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/djl/Documents/GitHub/Experiment/psychopyIOR/IOR3_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('warning')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('nextButton') is None:
        # initialise nextButton
        nextButton = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='nextButton',
        )
    if deviceManager.getDevice('key_instruct_3') is None:
        # initialise key_instruct_3
        key_instruct_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_3',
        )
    if deviceManager.getDevice('practice_key_resp') is None:
        # initialise practice_key_resp
        practice_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='practice_key_resp',
        )
    if deviceManager.getDevice('key_instruct') is None:
        # initialise key_instruct
        key_instruct = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_instruct_2') is None:
        # initialise key_instruct_2
        key_instruct_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "preCue_Instruction" ---
    text_instr_2 = visual.TextStim(win=win, name='text_instr_2',
        text='Focus your eyes on the plus sign at all times during the experiment.\n\nTwo circles will appear one after the other:\nThe first circle is the cue—please ignore it.\nThe second circle is the target—this is what you need to respond to.\n\nYour task:\nPress the left arrow key if the target appears in the left box.\nPress the right arrow key if the target appears in the right box.',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    nextButton = keyboard.Keyboard(deviceName='nextButton')
    precue_bottom_txt = visual.TextStim(win=win, name='precue_bottom_txt',
        text='Press Space to Continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Pre_Practice_window" ---
    text_norm_3 = visual.TextStim(win=win, name='text_norm_3',
        text='In the next portion you will get 15 practice trials!\n\nRemember to:\nOnly respond to the second circle \nKeep your eyes on the plus sign.',
        font='Arial',
        units='norm', pos=(0, .25), draggable=False, height=0.1, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_3 = keyboard.Keyboard(deviceName='key_instruct_3')
    preExperimentTxt_2 = visual.TextStim(win=win, name='preExperimentTxt_2',
        text='Press either arrow to Start',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "practice_trials" ---
    practice_fixation = visual.TextStim(win=win, name='practice_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    practice_leftBox = visual.Rect(
        win=win, name='practice_leftBox',units='cm', 
        width=(2.128, 2.128)[0], height=(2.128, 2.128)[1],
        ori=0.0, pos=(-5.334, 0), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-2.0, interpolate=True)
    practice_RightBox = visual.Rect(
        win=win, name='practice_RightBox',units='cm', 
        width=(2.128, 2.128)[0], height=(2.128, 2.128)[1],
        ori=0.0, pos=(5.334, 0), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-3.0, interpolate=True)
    practice_key_resp = keyboard.Keyboard(deviceName='practice_key_resp')
    practice_Cue = visual.Rect(
        win=win, name='practice_Cue',units='deg', 
        width=[2.128,2.128][0], height=[2.128,2.128][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-5.0, interpolate=True)
    practice_target = visual.Rect(
        win=win, name='practice_target',units='cm', 
        width=[2.128,2.128][0], height=[2.128,2.128][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-6.0, interpolate=True)
    
    # --- Initialize components for Routine "practice_error_feedback" ---
    # Run 'Begin Experiment' code from code_3
    msg = ''  # Default to an empty string
    
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Arial',
        pos=(0, .25), draggable=False, height=0.075, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Pre_experiment_window" ---
    text_norm = visual.TextStim(win=win, name='text_norm',
        text='In the next portion you will begin the experiment\nEvery 128 trials you will get a break, after 4 total blocks the experiment will close out. \nRemember to:\nOnly respond to the second circle (target), and keep your eyes on the plus sign.',
        font='Arial',
        units='norm', pos=(0, .25), draggable=False, height=0.1, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct = keyboard.Keyboard(deviceName='key_instruct')
    # Run 'Begin Experiment' code from text_align
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    text_norm.alignText= 'left'
    preExperimentTxt = visual.TextStim(win=win, name='preExperimentTxt',
        text='Press either arrow to Continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "trials" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    leftBox = visual.Rect(
        win=win, name='leftBox',units='cm', 
        width=(2.128, 2.128)[0], height=(2.128, 2.128)[1],
        ori=0.0, pos=(-5.334, 0), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-2.0, interpolate=True)
    rightBox = visual.Rect(
        win=win, name='rightBox',units='cm', 
        width=(2.128, 2.128)[0], height=(2.128, 2.128)[1],
        ori=0.0, pos=(5.334, 0), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-3.0, interpolate=True)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    cue = visual.Rect(
        win=win, name='cue',units='deg', 
        width=[2.128,2.128][0], height=[2.128,2.128][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-5.0, interpolate=True)
    target = visual.Rect(
        win=win, name='target',units='cm', 
        width=[2.128,2.128][0], height=[2.128,2.128][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-6.0, interpolate=True)
    
    # --- Initialize components for Routine "error_feedback" ---
    # Run 'Begin Experiment' code from code_2
    msg = ''  # Default to an empty string
    
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, .25), draggable=False, height=0.075, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "break_2" ---
    text_norm_2 = visual.TextStim(win=win, name='text_norm_2',
        text='You can take a short break now.\n\n\n\nPress any arrow to continue',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_2 = keyboard.Keyboard(deviceName='key_instruct_2')
    
    # --- Initialize components for Routine "trials" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    leftBox = visual.Rect(
        win=win, name='leftBox',units='cm', 
        width=(2.128, 2.128)[0], height=(2.128, 2.128)[1],
        ori=0.0, pos=(-5.334, 0), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-2.0, interpolate=True)
    rightBox = visual.Rect(
        win=win, name='rightBox',units='cm', 
        width=(2.128, 2.128)[0], height=(2.128, 2.128)[1],
        ori=0.0, pos=(5.334, 0), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-3.0, interpolate=True)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    cue = visual.Rect(
        win=win, name='cue',units='deg', 
        width=[2.128,2.128][0], height=[2.128,2.128][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-5.0, interpolate=True)
    target = visual.Rect(
        win=win, name='target',units='cm', 
        width=[2.128,2.128][0], height=[2.128,2.128][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-6.0, interpolate=True)
    
    # --- Initialize components for Routine "error_feedback" ---
    # Run 'Begin Experiment' code from code_2
    msg = ''  # Default to an empty string
    
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, .25), draggable=False, height=0.075, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "break_2" ---
    text_norm_2 = visual.TextStim(win=win, name='text_norm_2',
        text='You can take a short break now.\n\n\n\nPress any arrow to continue',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_2 = keyboard.Keyboard(deviceName='key_instruct_2')
    
    # --- Initialize components for Routine "trials" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    leftBox = visual.Rect(
        win=win, name='leftBox',units='cm', 
        width=(2.128, 2.128)[0], height=(2.128, 2.128)[1],
        ori=0.0, pos=(-5.334, 0), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-2.0, interpolate=True)
    rightBox = visual.Rect(
        win=win, name='rightBox',units='cm', 
        width=(2.128, 2.128)[0], height=(2.128, 2.128)[1],
        ori=0.0, pos=(5.334, 0), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-3.0, interpolate=True)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    cue = visual.Rect(
        win=win, name='cue',units='deg', 
        width=[2.128,2.128][0], height=[2.128,2.128][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-5.0, interpolate=True)
    target = visual.Rect(
        win=win, name='target',units='cm', 
        width=[2.128,2.128][0], height=[2.128,2.128][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-6.0, interpolate=True)
    
    # --- Initialize components for Routine "error_feedback" ---
    # Run 'Begin Experiment' code from code_2
    msg = ''  # Default to an empty string
    
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, .25), draggable=False, height=0.075, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "break_2" ---
    text_norm_2 = visual.TextStim(win=win, name='text_norm_2',
        text='You can take a short break now.\n\n\n\nPress any arrow to continue',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_2 = keyboard.Keyboard(deviceName='key_instruct_2')
    
    # --- Initialize components for Routine "trials" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    leftBox = visual.Rect(
        win=win, name='leftBox',units='cm', 
        width=(2.128, 2.128)[0], height=(2.128, 2.128)[1],
        ori=0.0, pos=(-5.334, 0), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-2.0, interpolate=True)
    rightBox = visual.Rect(
        win=win, name='rightBox',units='cm', 
        width=(2.128, 2.128)[0], height=(2.128, 2.128)[1],
        ori=0.0, pos=(5.334, 0), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-3.0, interpolate=True)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    cue = visual.Rect(
        win=win, name='cue',units='deg', 
        width=[2.128,2.128][0], height=[2.128,2.128][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-5.0, interpolate=True)
    target = visual.Rect(
        win=win, name='target',units='cm', 
        width=[2.128,2.128][0], height=[2.128,2.128][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-6.0, interpolate=True)
    
    # --- Initialize components for Routine "error_feedback" ---
    # Run 'Begin Experiment' code from code_2
    msg = ''  # Default to an empty string
    
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, .25), draggable=False, height=0.075, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "preCue_Instruction" ---
    # create an object to store info about Routine preCue_Instruction
    preCue_Instruction = data.Routine(
        name='preCue_Instruction',
        components=[text_instr_2, nextButton, precue_bottom_txt],
    )
    preCue_Instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for nextButton
    nextButton.keys = []
    nextButton.rt = []
    _nextButton_allKeys = []
    # store start times for preCue_Instruction
    preCue_Instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    preCue_Instruction.tStart = globalClock.getTime(format='float')
    preCue_Instruction.status = STARTED
    thisExp.addData('preCue_Instruction.started', preCue_Instruction.tStart)
    preCue_Instruction.maxDuration = None
    win.color = [-1.0000, -1.0000, -1.0000]
    win.colorSpace = 'rgb'
    win.backgroundImage = ''
    win.backgroundFit = 'none'
    # keep track of which components have finished
    preCue_InstructionComponents = preCue_Instruction.components
    for thisComponent in preCue_Instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "preCue_Instruction" ---
    preCue_Instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instr_2* updates
        
        # if text_instr_2 is starting this frame...
        if text_instr_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instr_2.frameNStart = frameN  # exact frame index
            text_instr_2.tStart = t  # local t and not account for scr refresh
            text_instr_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instr_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instr_2.started')
            # update status
            text_instr_2.status = STARTED
            text_instr_2.setAutoDraw(True)
        
        # if text_instr_2 is active this frame...
        if text_instr_2.status == STARTED:
            # update params
            pass
        
        # *nextButton* updates
        waitOnFlip = False
        
        # if nextButton is starting this frame...
        if nextButton.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            nextButton.frameNStart = frameN  # exact frame index
            nextButton.tStart = t  # local t and not account for scr refresh
            nextButton.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(nextButton, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'nextButton.started')
            # update status
            nextButton.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(nextButton.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(nextButton.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if nextButton.status == STARTED and not waitOnFlip:
            theseKeys = nextButton.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _nextButton_allKeys.extend(theseKeys)
            if len(_nextButton_allKeys):
                nextButton.keys = _nextButton_allKeys[0].name  # just the first key pressed
                nextButton.rt = _nextButton_allKeys[0].rt
                nextButton.duration = _nextButton_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # *precue_bottom_txt* updates
        
        # if precue_bottom_txt is starting this frame...
        if precue_bottom_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            precue_bottom_txt.frameNStart = frameN  # exact frame index
            precue_bottom_txt.tStart = t  # local t and not account for scr refresh
            precue_bottom_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(precue_bottom_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'precue_bottom_txt.started')
            # update status
            precue_bottom_txt.status = STARTED
            precue_bottom_txt.setAutoDraw(True)
        
        # if precue_bottom_txt is active this frame...
        if precue_bottom_txt.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            preCue_Instruction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in preCue_Instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "preCue_Instruction" ---
    for thisComponent in preCue_Instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for preCue_Instruction
    preCue_Instruction.tStop = globalClock.getTime(format='float')
    preCue_Instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('preCue_Instruction.stopped', preCue_Instruction.tStop)
    setupWindow(expInfo=expInfo, win=win)
    # check responses
    if nextButton.keys in ['', [], None]:  # No response was made
        nextButton.keys = None
    thisExp.addData('nextButton.keys',nextButton.keys)
    if nextButton.keys != None:  # we had a response
        thisExp.addData('nextButton.rt', nextButton.rt)
        thisExp.addData('nextButton.duration', nextButton.duration)
    thisExp.nextEntry()
    # the Routine "preCue_Instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Pre_Practice_window" ---
    # create an object to store info about Routine Pre_Practice_window
    Pre_Practice_window = data.Routine(
        name='Pre_Practice_window',
        components=[text_norm_3, key_instruct_3, preExperimentTxt_2],
    )
    Pre_Practice_window.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_3
    key_instruct_3.keys = []
    key_instruct_3.rt = []
    _key_instruct_3_allKeys = []
    # store start times for Pre_Practice_window
    Pre_Practice_window.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Pre_Practice_window.tStart = globalClock.getTime(format='float')
    Pre_Practice_window.status = STARTED
    thisExp.addData('Pre_Practice_window.started', Pre_Practice_window.tStart)
    Pre_Practice_window.maxDuration = None
    win.color = [-1.0000, -1.0000, -1.0000]
    win.colorSpace = 'rgb'
    win.backgroundImage = ''
    win.backgroundFit = 'none'
    # keep track of which components have finished
    Pre_Practice_windowComponents = Pre_Practice_window.components
    for thisComponent in Pre_Practice_window.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Pre_Practice_window" ---
    Pre_Practice_window.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm_3* updates
        
        # if text_norm_3 is starting this frame...
        if text_norm_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm_3.frameNStart = frameN  # exact frame index
            text_norm_3.tStart = t  # local t and not account for scr refresh
            text_norm_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm_3.status = STARTED
            text_norm_3.setAutoDraw(True)
        
        # if text_norm_3 is active this frame...
        if text_norm_3.status == STARTED:
            # update params
            pass
        
        # *key_instruct_3* updates
        waitOnFlip = False
        
        # if key_instruct_3 is starting this frame...
        if key_instruct_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_3.frameNStart = frameN  # exact frame index
            key_instruct_3.tStart = t  # local t and not account for scr refresh
            key_instruct_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct_3.started')
            # update status
            key_instruct_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_3.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_3.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_3_allKeys.extend(theseKeys)
            if len(_key_instruct_3_allKeys):
                key_instruct_3.keys = _key_instruct_3_allKeys[0].name  # just the first key pressed
                key_instruct_3.rt = _key_instruct_3_allKeys[0].rt
                key_instruct_3.duration = _key_instruct_3_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # *preExperimentTxt_2* updates
        
        # if preExperimentTxt_2 is starting this frame...
        if preExperimentTxt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            preExperimentTxt_2.frameNStart = frameN  # exact frame index
            preExperimentTxt_2.tStart = t  # local t and not account for scr refresh
            preExperimentTxt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(preExperimentTxt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'preExperimentTxt_2.started')
            # update status
            preExperimentTxt_2.status = STARTED
            preExperimentTxt_2.setAutoDraw(True)
        
        # if preExperimentTxt_2 is active this frame...
        if preExperimentTxt_2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Pre_Practice_window.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Pre_Practice_window.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Pre_Practice_window" ---
    for thisComponent in Pre_Practice_window.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Pre_Practice_window
    Pre_Practice_window.tStop = globalClock.getTime(format='float')
    Pre_Practice_window.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Pre_Practice_window.stopped', Pre_Practice_window.tStop)
    setupWindow(expInfo=expInfo, win=win)
    # check responses
    if key_instruct_3.keys in ['', [], None]:  # No response was made
        key_instruct_3.keys = None
    thisExp.addData('key_instruct_3.keys',key_instruct_3.keys)
    if key_instruct_3.keys != None:  # we had a response
        thisExp.addData('key_instruct_3.rt', key_instruct_3.rt)
        thisExp.addData('key_instruct_3.duration', key_instruct_3.duration)
    thisExp.nextEntry()
    # the Routine "Pre_Practice_window" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_loop = data.TrialHandler2(
        name='practice_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('practiceconditions.csv'), 
        seed=None, 
    )
    thisExp.addLoop(practice_loop)  # add the loop to the experiment
    thisPractice_loop = practice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
    if thisPractice_loop != None:
        for paramName in thisPractice_loop:
            globals()[paramName] = thisPractice_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_loop in practice_loop:
        currentLoop = practice_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
        if thisPractice_loop != None:
            for paramName in thisPractice_loop:
                globals()[paramName] = thisPractice_loop[paramName]
        
        # --- Prepare to start Routine "practice_trials" ---
        # create an object to store info about Routine practice_trials
        practice_trials = data.Routine(
            name='practice_trials',
            components=[practice_fixation, practice_leftBox, practice_RightBox, practice_key_resp, practice_Cue, practice_target],
        )
        practice_trials.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for practice_key_resp
        practice_key_resp.keys = []
        practice_key_resp.rt = []
        _practice_key_resp_allKeys = []
        practice_Cue.setPos(cueSide)
        practice_target.setPos(targetSide)
        # store start times for practice_trials
        practice_trials.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_trials.tStart = globalClock.getTime(format='float')
        practice_trials.status = STARTED
        thisExp.addData('practice_trials.started', practice_trials.tStart)
        practice_trials.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'fill'
        # keep track of which components have finished
        practice_trialsComponents = practice_trials.components
        for thisComponent in practice_trials.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_trials" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        practice_trials.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_fixation* updates
            
            # if practice_fixation is starting this frame...
            if practice_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_fixation.frameNStart = frameN  # exact frame index
                practice_fixation.tStart = t  # local t and not account for scr refresh
                practice_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_fixation.started')
                # update status
                practice_fixation.status = STARTED
                practice_fixation.setAutoDraw(True)
            
            # if practice_fixation is active this frame...
            if practice_fixation.status == STARTED:
                # update params
                pass
            
            # if practice_fixation is stopping this frame...
            if practice_fixation.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 0.8+ISI+ 1-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_fixation.tStop = t  # not accounting for scr refresh
                    practice_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_fixation.stopped')
                    # update status
                    practice_fixation.status = FINISHED
                    practice_fixation.setAutoDraw(False)
            
            # *practice_leftBox* updates
            
            # if practice_leftBox is starting this frame...
            if practice_leftBox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_leftBox.frameNStart = frameN  # exact frame index
                practice_leftBox.tStart = t  # local t and not account for scr refresh
                practice_leftBox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_leftBox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_leftBox.started')
                # update status
                practice_leftBox.status = STARTED
                practice_leftBox.setAutoDraw(True)
            
            # if practice_leftBox is active this frame...
            if practice_leftBox.status == STARTED:
                # update params
                practice_leftBox.setLineColor([-1.0000, -1.0000, -1.0000], log=False)
            
            # if practice_leftBox is stopping this frame...
            if practice_leftBox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_leftBox.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_leftBox.tStop = t  # not accounting for scr refresh
                    practice_leftBox.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_leftBox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_leftBox.stopped')
                    # update status
                    practice_leftBox.status = FINISHED
                    practice_leftBox.setAutoDraw(False)
            
            # *practice_RightBox* updates
            
            # if practice_RightBox is starting this frame...
            if practice_RightBox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_RightBox.frameNStart = frameN  # exact frame index
                practice_RightBox.tStart = t  # local t and not account for scr refresh
                practice_RightBox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_RightBox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_RightBox.started')
                # update status
                practice_RightBox.status = STARTED
                practice_RightBox.setAutoDraw(True)
            
            # if practice_RightBox is active this frame...
            if practice_RightBox.status == STARTED:
                # update params
                practice_RightBox.setLineColor([-1.0000, -1.0000, -1.0000], log=False)
            
            # if practice_RightBox is stopping this frame...
            if practice_RightBox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_RightBox.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_RightBox.tStop = t  # not accounting for scr refresh
                    practice_RightBox.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_RightBox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_RightBox.stopped')
                    # update status
                    practice_RightBox.status = FINISHED
                    practice_RightBox.setAutoDraw(False)
            
            # *practice_key_resp* updates
            waitOnFlip = False
            
            # if practice_key_resp is starting this frame...
            if practice_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_key_resp.frameNStart = frameN  # exact frame index
                practice_key_resp.tStart = t  # local t and not account for scr refresh
                practice_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_key_resp.started')
                # update status
                practice_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(practice_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(practice_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if practice_key_resp is stopping this frame...
            if practice_key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_key_resp.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_key_resp.tStop = t  # not accounting for scr refresh
                    practice_key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_key_resp.stopped')
                    # update status
                    practice_key_resp.status = FINISHED
                    practice_key_resp.status = FINISHED
            if practice_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = practice_key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _practice_key_resp_allKeys.extend(theseKeys)
                if len(_practice_key_resp_allKeys):
                    practice_key_resp.keys = _practice_key_resp_allKeys[0].name  # just the first key pressed
                    practice_key_resp.rt = _practice_key_resp_allKeys[0].rt
                    practice_key_resp.duration = _practice_key_resp_allKeys[0].duration
                    # was this correct?
                    if (practice_key_resp.keys == str(correct)) or (practice_key_resp.keys == correct):
                        practice_key_resp.corr = 1
                    else:
                        practice_key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *practice_Cue* updates
            
            # if practice_Cue is starting this frame...
            if practice_Cue.status == NOT_STARTED and tThisFlip >= .75-frameTolerance:
                # keep track of start time/frame for later
                practice_Cue.frameNStart = frameN  # exact frame index
                practice_Cue.tStart = t  # local t and not account for scr refresh
                practice_Cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_Cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_Cue.started')
                # update status
                practice_Cue.status = STARTED
                practice_Cue.setAutoDraw(True)
            
            # if practice_Cue is active this frame...
            if practice_Cue.status == STARTED:
                # update params
                pass
            
            # if practice_Cue is stopping this frame...
            if practice_Cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_Cue.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_Cue.tStop = t  # not accounting for scr refresh
                    practice_Cue.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_Cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_Cue.stopped')
                    # update status
                    practice_Cue.status = FINISHED
                    practice_Cue.setAutoDraw(False)
            
            # *practice_target* updates
            
            # if practice_target is starting this frame...
            if practice_target.status == NOT_STARTED and tThisFlip >= 0.8+ISI-frameTolerance:
                # keep track of start time/frame for later
                practice_target.frameNStart = frameN  # exact frame index
                practice_target.tStart = t  # local t and not account for scr refresh
                practice_target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_target.started')
                # update status
                practice_target.status = STARTED
                practice_target.setAutoDraw(True)
            
            # if practice_target is active this frame...
            if practice_target.status == STARTED:
                # update params
                pass
            
            # if practice_target is stopping this frame...
            if practice_target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_target.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_target.tStop = t  # not accounting for scr refresh
                    practice_target.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_target.stopped')
                    # update status
                    practice_target.status = FINISHED
                    practice_target.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                practice_trials.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_trials.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_trials" ---
        for thisComponent in practice_trials.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_trials
        practice_trials.tStop = globalClock.getTime(format='float')
        practice_trials.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_trials.stopped', practice_trials.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # check responses
        if practice_key_resp.keys in ['', [], None]:  # No response was made
            practice_key_resp.keys = None
            # was no response the correct answer?!
            if str(correct).lower() == 'none':
               practice_key_resp.corr = 1;  # correct non-response
            else:
               practice_key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for practice_loop (TrialHandler)
        practice_loop.addData('practice_key_resp.keys',practice_key_resp.keys)
        practice_loop.addData('practice_key_resp.corr', practice_key_resp.corr)
        if practice_key_resp.keys != None:  # we had a response
            practice_loop.addData('practice_key_resp.rt', practice_key_resp.rt)
            practice_loop.addData('practice_key_resp.duration', practice_key_resp.duration)
        # the Routine "practice_trials" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "practice_error_feedback" ---
        # create an object to store info about Routine practice_error_feedback
        practice_error_feedback = data.Routine(
            name='practice_error_feedback',
            components=[text_2],
        )
        practice_error_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_3
        if not practice_key_resp.keys:  # No response was made
            msg = 'Too slow! Please respond faster.'
        else:
            msg = ''
        
        
        text_2.setColor('red', colorSpace='rgb')
        text_2.setText(msg)
        # store start times for practice_error_feedback
        practice_error_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_error_feedback.tStart = globalClock.getTime(format='float')
        practice_error_feedback.status = STARTED
        thisExp.addData('practice_error_feedback.started', practice_error_feedback.tStart)
        practice_error_feedback.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        practice_error_feedbackComponents = practice_error_feedback.components
        for thisComponent in practice_error_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_error_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        practice_error_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # if text_2 is stopping this frame...
            if text_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    text_2.tStop = t  # not accounting for scr refresh
                    text_2.tStopRefresh = tThisFlipGlobal  # on global time
                    text_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_2.stopped')
                    # update status
                    text_2.status = FINISHED
                    text_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                practice_error_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_error_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_error_feedback" ---
        for thisComponent in practice_error_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_error_feedback
        practice_error_feedback.tStop = globalClock.getTime(format='float')
        practice_error_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_error_feedback.stopped', practice_error_feedback.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_error_feedback.maxDurationReached:
            routineTimer.addTime(-practice_error_feedback.maxDuration)
        elif practice_error_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practice_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Pre_experiment_window" ---
    # create an object to store info about Routine Pre_experiment_window
    Pre_experiment_window = data.Routine(
        name='Pre_experiment_window',
        components=[text_norm, key_instruct, preExperimentTxt],
    )
    Pre_experiment_window.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct
    key_instruct.keys = []
    key_instruct.rt = []
    _key_instruct_allKeys = []
    # store start times for Pre_experiment_window
    Pre_experiment_window.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Pre_experiment_window.tStart = globalClock.getTime(format='float')
    Pre_experiment_window.status = STARTED
    thisExp.addData('Pre_experiment_window.started', Pre_experiment_window.tStart)
    Pre_experiment_window.maxDuration = None
    win.color = [-1.0000, -1.0000, -1.0000]
    win.colorSpace = 'rgb'
    win.backgroundImage = ''
    win.backgroundFit = 'none'
    # keep track of which components have finished
    Pre_experiment_windowComponents = Pre_experiment_window.components
    for thisComponent in Pre_experiment_window.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Pre_experiment_window" ---
    Pre_experiment_window.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm* updates
        
        # if text_norm is starting this frame...
        if text_norm.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm.frameNStart = frameN  # exact frame index
            text_norm.tStart = t  # local t and not account for scr refresh
            text_norm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm.status = STARTED
            text_norm.setAutoDraw(True)
        
        # if text_norm is active this frame...
        if text_norm.status == STARTED:
            # update params
            pass
        
        # *key_instruct* updates
        waitOnFlip = False
        
        # if key_instruct is starting this frame...
        if key_instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct.frameNStart = frameN  # exact frame index
            key_instruct.tStart = t  # local t and not account for scr refresh
            key_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct.started')
            # update status
            key_instruct.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_allKeys.extend(theseKeys)
            if len(_key_instruct_allKeys):
                key_instruct.keys = _key_instruct_allKeys[0].name  # just the first key pressed
                key_instruct.rt = _key_instruct_allKeys[0].rt
                key_instruct.duration = _key_instruct_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # *preExperimentTxt* updates
        
        # if preExperimentTxt is starting this frame...
        if preExperimentTxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            preExperimentTxt.frameNStart = frameN  # exact frame index
            preExperimentTxt.tStart = t  # local t and not account for scr refresh
            preExperimentTxt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(preExperimentTxt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'preExperimentTxt.started')
            # update status
            preExperimentTxt.status = STARTED
            preExperimentTxt.setAutoDraw(True)
        
        # if preExperimentTxt is active this frame...
        if preExperimentTxt.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Pre_experiment_window.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Pre_experiment_window.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Pre_experiment_window" ---
    for thisComponent in Pre_experiment_window.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Pre_experiment_window
    Pre_experiment_window.tStop = globalClock.getTime(format='float')
    Pre_experiment_window.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Pre_experiment_window.stopped', Pre_experiment_window.tStop)
    setupWindow(expInfo=expInfo, win=win)
    # check responses
    if key_instruct.keys in ['', [], None]:  # No response was made
        key_instruct.keys = None
    thisExp.addData('key_instruct.keys',key_instruct.keys)
    if key_instruct.keys != None:  # we had a response
        thisExp.addData('key_instruct.rt', key_instruct.rt)
        thisExp.addData('key_instruct.duration', key_instruct.duration)
    thisExp.nextEntry()
    # the Routine "Pre_experiment_window" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_loop = data.TrialHandler2(
        name='trial_loop',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trial_loop)  # add the loop to the experiment
    thisTrial_loop = trial_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
    if thisTrial_loop != None:
        for paramName in thisTrial_loop:
            globals()[paramName] = thisTrial_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_loop in trial_loop:
        currentLoop = trial_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
        if thisTrial_loop != None:
            for paramName in thisTrial_loop:
                globals()[paramName] = thisTrial_loop[paramName]
        
        # --- Prepare to start Routine "trials" ---
        # create an object to store info about Routine trials
        trials = data.Routine(
            name='trials',
            components=[fixation, leftBox, rightBox, key_resp, cue, target],
        )
        trials.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        cue.setPos(cueSide)
        target.setPos(targetSide)
        # store start times for trials
        trials.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trials.tStart = globalClock.getTime(format='float')
        trials.status = STARTED
        thisExp.addData('trials.started', trials.tStart)
        trials.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        trialsComponents = trials.components
        for thisComponent in trials.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trials" ---
        # if trial has changed, end Routine now
        if isinstance(trial_loop, data.TrialHandler2) and thisTrial_loop.thisN != trial_loop.thisTrial.thisN:
            continueRoutine = False
        trials.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 0.8+ISI+ 1-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *leftBox* updates
            
            # if leftBox is starting this frame...
            if leftBox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                leftBox.frameNStart = frameN  # exact frame index
                leftBox.tStart = t  # local t and not account for scr refresh
                leftBox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(leftBox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'leftBox.started')
                # update status
                leftBox.status = STARTED
                leftBox.setAutoDraw(True)
            
            # if leftBox is active this frame...
            if leftBox.status == STARTED:
                # update params
                leftBox.setLineColor([-1.0000, -1.0000, -1.0000], log=False)
            
            # if leftBox is stopping this frame...
            if leftBox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > leftBox.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    leftBox.tStop = t  # not accounting for scr refresh
                    leftBox.tStopRefresh = tThisFlipGlobal  # on global time
                    leftBox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'leftBox.stopped')
                    # update status
                    leftBox.status = FINISHED
                    leftBox.setAutoDraw(False)
            
            # *rightBox* updates
            
            # if rightBox is starting this frame...
            if rightBox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rightBox.frameNStart = frameN  # exact frame index
                rightBox.tStart = t  # local t and not account for scr refresh
                rightBox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rightBox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rightBox.started')
                # update status
                rightBox.status = STARTED
                rightBox.setAutoDraw(True)
            
            # if rightBox is active this frame...
            if rightBox.status == STARTED:
                # update params
                rightBox.setLineColor([-1.0000, -1.0000, -1.0000], log=False)
            
            # if rightBox is stopping this frame...
            if rightBox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rightBox.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    rightBox.tStop = t  # not accounting for scr refresh
                    rightBox.tStopRefresh = tThisFlipGlobal  # on global time
                    rightBox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rightBox.stopped')
                    # update status
                    rightBox.status = FINISHED
                    rightBox.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # was this correct?
                    if (key_resp.keys == str(correct)) or (key_resp.keys == correct):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *cue* updates
            
            # if cue is starting this frame...
            if cue.status == NOT_STARTED and tThisFlip >= .75-frameTolerance:
                # keep track of start time/frame for later
                cue.frameNStart = frameN  # exact frame index
                cue.tStart = t  # local t and not account for scr refresh
                cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue.started')
                # update status
                cue.status = STARTED
                cue.setAutoDraw(True)
            
            # if cue is active this frame...
            if cue.status == STARTED:
                # update params
                pass
            
            # if cue is stopping this frame...
            if cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    cue.tStop = t  # not accounting for scr refresh
                    cue.tStopRefresh = tThisFlipGlobal  # on global time
                    cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue.stopped')
                    # update status
                    cue.status = FINISHED
                    cue.setAutoDraw(False)
            
            # *target* updates
            
            # if target is starting this frame...
            if target.status == NOT_STARTED and tThisFlip >= 0.8+ISI-frameTolerance:
                # keep track of start time/frame for later
                target.frameNStart = frameN  # exact frame index
                target.tStart = t  # local t and not account for scr refresh
                target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target.started')
                # update status
                target.status = STARTED
                target.setAutoDraw(True)
            
            # if target is active this frame...
            if target.status == STARTED:
                # update params
                pass
            
            # if target is stopping this frame...
            if target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    target.tStop = t  # not accounting for scr refresh
                    target.tStopRefresh = tThisFlipGlobal  # on global time
                    target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target.stopped')
                    # update status
                    target.status = FINISHED
                    target.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trials.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trials.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trials" ---
        for thisComponent in trials.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trials
        trials.tStop = globalClock.getTime(format='float')
        trials.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trials.stopped', trials.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(correct).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trial_loop (TrialHandler)
        trial_loop.addData('key_resp.keys',key_resp.keys)
        trial_loop.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            trial_loop.addData('key_resp.rt', key_resp.rt)
            trial_loop.addData('key_resp.duration', key_resp.duration)
        # the Routine "trials" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "error_feedback" ---
        # create an object to store info about Routine error_feedback
        error_feedback = data.Routine(
            name='error_feedback',
            components=[text],
        )
        error_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        if not key_resp.keys:  # No response was made
            msg = 'Too slow! Please respond faster.'
        else:
            msg = ''
        
        
        text.setColor([1.0000, -1.0000, -1.0000], colorSpace='rgb')
        text.setText(msg)
        # store start times for error_feedback
        error_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        error_feedback.tStart = globalClock.getTime(format='float')
        error_feedback.status = STARTED
        thisExp.addData('error_feedback.started', error_feedback.tStart)
        error_feedback.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        error_feedbackComponents = error_feedback.components
        for thisComponent in error_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "error_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trial_loop, data.TrialHandler2) and thisTrial_loop.thisN != trial_loop.thisTrial.thisN:
            continueRoutine = False
        error_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                error_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in error_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "error_feedback" ---
        for thisComponent in error_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for error_feedback
        error_feedback.tStop = globalClock.getTime(format='float')
        error_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('error_feedback.stopped', error_feedback.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if error_feedback.maxDurationReached:
            routineTimer.addTime(-error_feedback.maxDuration)
        elif error_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'trial_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[text_norm_2, key_instruct_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_2
    key_instruct_2.keys = []
    key_instruct_2.rt = []
    _key_instruct_2_allKeys = []
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm_2* updates
        
        # if text_norm_2 is starting this frame...
        if text_norm_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm_2.frameNStart = frameN  # exact frame index
            text_norm_2.tStart = t  # local t and not account for scr refresh
            text_norm_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm_2.status = STARTED
            text_norm_2.setAutoDraw(True)
        
        # if text_norm_2 is active this frame...
        if text_norm_2.status == STARTED:
            # update params
            pass
        
        # *key_instruct_2* updates
        waitOnFlip = False
        
        # if key_instruct_2 is starting this frame...
        if key_instruct_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_2.frameNStart = frameN  # exact frame index
            key_instruct_2.tStart = t  # local t and not account for scr refresh
            key_instruct_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct_2.started')
            # update status
            key_instruct_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_2.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_2.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_2_allKeys.extend(theseKeys)
            if len(_key_instruct_2_allKeys):
                key_instruct_2.keys = _key_instruct_2_allKeys[0].name  # just the first key pressed
                key_instruct_2.rt = _key_instruct_2_allKeys[0].rt
                key_instruct_2.duration = _key_instruct_2_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # check responses
    if key_instruct_2.keys in ['', [], None]:  # No response was made
        key_instruct_2.keys = None
    thisExp.addData('key_instruct_2.keys',key_instruct_2.keys)
    if key_instruct_2.keys != None:  # we had a response
        thisExp.addData('key_instruct_2.rt', key_instruct_2.rt)
        thisExp.addData('key_instruct_2.duration', key_instruct_2.duration)
    thisExp.nextEntry()
    # the Routine "break_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_loop2 = data.TrialHandler2(
        name='trial_loop2',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trial_loop2)  # add the loop to the experiment
    thisTrial_loop2 = trial_loop2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop2.rgb)
    if thisTrial_loop2 != None:
        for paramName in thisTrial_loop2:
            globals()[paramName] = thisTrial_loop2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_loop2 in trial_loop2:
        currentLoop = trial_loop2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop2.rgb)
        if thisTrial_loop2 != None:
            for paramName in thisTrial_loop2:
                globals()[paramName] = thisTrial_loop2[paramName]
        
        # --- Prepare to start Routine "trials" ---
        # create an object to store info about Routine trials
        trials = data.Routine(
            name='trials',
            components=[fixation, leftBox, rightBox, key_resp, cue, target],
        )
        trials.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        cue.setPos(cueSide)
        target.setPos(targetSide)
        # store start times for trials
        trials.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trials.tStart = globalClock.getTime(format='float')
        trials.status = STARTED
        thisExp.addData('trials.started', trials.tStart)
        trials.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        trialsComponents = trials.components
        for thisComponent in trials.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trials" ---
        # if trial has changed, end Routine now
        if isinstance(trial_loop2, data.TrialHandler2) and thisTrial_loop2.thisN != trial_loop2.thisTrial.thisN:
            continueRoutine = False
        trials.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 0.8+ISI+ 1-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *leftBox* updates
            
            # if leftBox is starting this frame...
            if leftBox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                leftBox.frameNStart = frameN  # exact frame index
                leftBox.tStart = t  # local t and not account for scr refresh
                leftBox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(leftBox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'leftBox.started')
                # update status
                leftBox.status = STARTED
                leftBox.setAutoDraw(True)
            
            # if leftBox is active this frame...
            if leftBox.status == STARTED:
                # update params
                leftBox.setLineColor([-1.0000, -1.0000, -1.0000], log=False)
            
            # if leftBox is stopping this frame...
            if leftBox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > leftBox.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    leftBox.tStop = t  # not accounting for scr refresh
                    leftBox.tStopRefresh = tThisFlipGlobal  # on global time
                    leftBox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'leftBox.stopped')
                    # update status
                    leftBox.status = FINISHED
                    leftBox.setAutoDraw(False)
            
            # *rightBox* updates
            
            # if rightBox is starting this frame...
            if rightBox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rightBox.frameNStart = frameN  # exact frame index
                rightBox.tStart = t  # local t and not account for scr refresh
                rightBox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rightBox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rightBox.started')
                # update status
                rightBox.status = STARTED
                rightBox.setAutoDraw(True)
            
            # if rightBox is active this frame...
            if rightBox.status == STARTED:
                # update params
                rightBox.setLineColor([-1.0000, -1.0000, -1.0000], log=False)
            
            # if rightBox is stopping this frame...
            if rightBox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rightBox.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    rightBox.tStop = t  # not accounting for scr refresh
                    rightBox.tStopRefresh = tThisFlipGlobal  # on global time
                    rightBox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rightBox.stopped')
                    # update status
                    rightBox.status = FINISHED
                    rightBox.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # was this correct?
                    if (key_resp.keys == str(correct)) or (key_resp.keys == correct):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *cue* updates
            
            # if cue is starting this frame...
            if cue.status == NOT_STARTED and tThisFlip >= .75-frameTolerance:
                # keep track of start time/frame for later
                cue.frameNStart = frameN  # exact frame index
                cue.tStart = t  # local t and not account for scr refresh
                cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue.started')
                # update status
                cue.status = STARTED
                cue.setAutoDraw(True)
            
            # if cue is active this frame...
            if cue.status == STARTED:
                # update params
                pass
            
            # if cue is stopping this frame...
            if cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    cue.tStop = t  # not accounting for scr refresh
                    cue.tStopRefresh = tThisFlipGlobal  # on global time
                    cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue.stopped')
                    # update status
                    cue.status = FINISHED
                    cue.setAutoDraw(False)
            
            # *target* updates
            
            # if target is starting this frame...
            if target.status == NOT_STARTED and tThisFlip >= 0.8+ISI-frameTolerance:
                # keep track of start time/frame for later
                target.frameNStart = frameN  # exact frame index
                target.tStart = t  # local t and not account for scr refresh
                target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target.started')
                # update status
                target.status = STARTED
                target.setAutoDraw(True)
            
            # if target is active this frame...
            if target.status == STARTED:
                # update params
                pass
            
            # if target is stopping this frame...
            if target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    target.tStop = t  # not accounting for scr refresh
                    target.tStopRefresh = tThisFlipGlobal  # on global time
                    target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target.stopped')
                    # update status
                    target.status = FINISHED
                    target.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trials.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trials.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trials" ---
        for thisComponent in trials.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trials
        trials.tStop = globalClock.getTime(format='float')
        trials.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trials.stopped', trials.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(correct).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trial_loop2 (TrialHandler)
        trial_loop2.addData('key_resp.keys',key_resp.keys)
        trial_loop2.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            trial_loop2.addData('key_resp.rt', key_resp.rt)
            trial_loop2.addData('key_resp.duration', key_resp.duration)
        # the Routine "trials" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "error_feedback" ---
        # create an object to store info about Routine error_feedback
        error_feedback = data.Routine(
            name='error_feedback',
            components=[text],
        )
        error_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        if not key_resp.keys:  # No response was made
            msg = 'Too slow! Please respond faster.'
        else:
            msg = ''
        
        
        text.setColor([1.0000, -1.0000, -1.0000], colorSpace='rgb')
        text.setText(msg)
        # store start times for error_feedback
        error_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        error_feedback.tStart = globalClock.getTime(format='float')
        error_feedback.status = STARTED
        thisExp.addData('error_feedback.started', error_feedback.tStart)
        error_feedback.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        error_feedbackComponents = error_feedback.components
        for thisComponent in error_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "error_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trial_loop2, data.TrialHandler2) and thisTrial_loop2.thisN != trial_loop2.thisTrial.thisN:
            continueRoutine = False
        error_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                error_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in error_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "error_feedback" ---
        for thisComponent in error_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for error_feedback
        error_feedback.tStop = globalClock.getTime(format='float')
        error_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('error_feedback.stopped', error_feedback.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if error_feedback.maxDurationReached:
            routineTimer.addTime(-error_feedback.maxDuration)
        elif error_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'trial_loop2'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[text_norm_2, key_instruct_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_2
    key_instruct_2.keys = []
    key_instruct_2.rt = []
    _key_instruct_2_allKeys = []
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm_2* updates
        
        # if text_norm_2 is starting this frame...
        if text_norm_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm_2.frameNStart = frameN  # exact frame index
            text_norm_2.tStart = t  # local t and not account for scr refresh
            text_norm_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm_2.status = STARTED
            text_norm_2.setAutoDraw(True)
        
        # if text_norm_2 is active this frame...
        if text_norm_2.status == STARTED:
            # update params
            pass
        
        # *key_instruct_2* updates
        waitOnFlip = False
        
        # if key_instruct_2 is starting this frame...
        if key_instruct_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_2.frameNStart = frameN  # exact frame index
            key_instruct_2.tStart = t  # local t and not account for scr refresh
            key_instruct_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct_2.started')
            # update status
            key_instruct_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_2.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_2.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_2_allKeys.extend(theseKeys)
            if len(_key_instruct_2_allKeys):
                key_instruct_2.keys = _key_instruct_2_allKeys[0].name  # just the first key pressed
                key_instruct_2.rt = _key_instruct_2_allKeys[0].rt
                key_instruct_2.duration = _key_instruct_2_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # check responses
    if key_instruct_2.keys in ['', [], None]:  # No response was made
        key_instruct_2.keys = None
    thisExp.addData('key_instruct_2.keys',key_instruct_2.keys)
    if key_instruct_2.keys != None:  # we had a response
        thisExp.addData('key_instruct_2.rt', key_instruct_2.rt)
        thisExp.addData('key_instruct_2.duration', key_instruct_2.duration)
    thisExp.nextEntry()
    # the Routine "break_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_loop3 = data.TrialHandler2(
        name='trial_loop3',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trial_loop3)  # add the loop to the experiment
    thisTrial_loop3 = trial_loop3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop3.rgb)
    if thisTrial_loop3 != None:
        for paramName in thisTrial_loop3:
            globals()[paramName] = thisTrial_loop3[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_loop3 in trial_loop3:
        currentLoop = trial_loop3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop3.rgb)
        if thisTrial_loop3 != None:
            for paramName in thisTrial_loop3:
                globals()[paramName] = thisTrial_loop3[paramName]
        
        # --- Prepare to start Routine "trials" ---
        # create an object to store info about Routine trials
        trials = data.Routine(
            name='trials',
            components=[fixation, leftBox, rightBox, key_resp, cue, target],
        )
        trials.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        cue.setPos(cueSide)
        target.setPos(targetSide)
        # store start times for trials
        trials.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trials.tStart = globalClock.getTime(format='float')
        trials.status = STARTED
        thisExp.addData('trials.started', trials.tStart)
        trials.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        trialsComponents = trials.components
        for thisComponent in trials.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trials" ---
        # if trial has changed, end Routine now
        if isinstance(trial_loop3, data.TrialHandler2) and thisTrial_loop3.thisN != trial_loop3.thisTrial.thisN:
            continueRoutine = False
        trials.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 0.8+ISI+ 1-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *leftBox* updates
            
            # if leftBox is starting this frame...
            if leftBox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                leftBox.frameNStart = frameN  # exact frame index
                leftBox.tStart = t  # local t and not account for scr refresh
                leftBox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(leftBox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'leftBox.started')
                # update status
                leftBox.status = STARTED
                leftBox.setAutoDraw(True)
            
            # if leftBox is active this frame...
            if leftBox.status == STARTED:
                # update params
                leftBox.setLineColor([-1.0000, -1.0000, -1.0000], log=False)
            
            # if leftBox is stopping this frame...
            if leftBox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > leftBox.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    leftBox.tStop = t  # not accounting for scr refresh
                    leftBox.tStopRefresh = tThisFlipGlobal  # on global time
                    leftBox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'leftBox.stopped')
                    # update status
                    leftBox.status = FINISHED
                    leftBox.setAutoDraw(False)
            
            # *rightBox* updates
            
            # if rightBox is starting this frame...
            if rightBox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rightBox.frameNStart = frameN  # exact frame index
                rightBox.tStart = t  # local t and not account for scr refresh
                rightBox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rightBox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rightBox.started')
                # update status
                rightBox.status = STARTED
                rightBox.setAutoDraw(True)
            
            # if rightBox is active this frame...
            if rightBox.status == STARTED:
                # update params
                rightBox.setLineColor([-1.0000, -1.0000, -1.0000], log=False)
            
            # if rightBox is stopping this frame...
            if rightBox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rightBox.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    rightBox.tStop = t  # not accounting for scr refresh
                    rightBox.tStopRefresh = tThisFlipGlobal  # on global time
                    rightBox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rightBox.stopped')
                    # update status
                    rightBox.status = FINISHED
                    rightBox.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # was this correct?
                    if (key_resp.keys == str(correct)) or (key_resp.keys == correct):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *cue* updates
            
            # if cue is starting this frame...
            if cue.status == NOT_STARTED and tThisFlip >= .75-frameTolerance:
                # keep track of start time/frame for later
                cue.frameNStart = frameN  # exact frame index
                cue.tStart = t  # local t and not account for scr refresh
                cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue.started')
                # update status
                cue.status = STARTED
                cue.setAutoDraw(True)
            
            # if cue is active this frame...
            if cue.status == STARTED:
                # update params
                pass
            
            # if cue is stopping this frame...
            if cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    cue.tStop = t  # not accounting for scr refresh
                    cue.tStopRefresh = tThisFlipGlobal  # on global time
                    cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue.stopped')
                    # update status
                    cue.status = FINISHED
                    cue.setAutoDraw(False)
            
            # *target* updates
            
            # if target is starting this frame...
            if target.status == NOT_STARTED and tThisFlip >= 0.8+ISI-frameTolerance:
                # keep track of start time/frame for later
                target.frameNStart = frameN  # exact frame index
                target.tStart = t  # local t and not account for scr refresh
                target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target.started')
                # update status
                target.status = STARTED
                target.setAutoDraw(True)
            
            # if target is active this frame...
            if target.status == STARTED:
                # update params
                pass
            
            # if target is stopping this frame...
            if target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    target.tStop = t  # not accounting for scr refresh
                    target.tStopRefresh = tThisFlipGlobal  # on global time
                    target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target.stopped')
                    # update status
                    target.status = FINISHED
                    target.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trials.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trials.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trials" ---
        for thisComponent in trials.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trials
        trials.tStop = globalClock.getTime(format='float')
        trials.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trials.stopped', trials.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(correct).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trial_loop3 (TrialHandler)
        trial_loop3.addData('key_resp.keys',key_resp.keys)
        trial_loop3.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            trial_loop3.addData('key_resp.rt', key_resp.rt)
            trial_loop3.addData('key_resp.duration', key_resp.duration)
        # the Routine "trials" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "error_feedback" ---
        # create an object to store info about Routine error_feedback
        error_feedback = data.Routine(
            name='error_feedback',
            components=[text],
        )
        error_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        if not key_resp.keys:  # No response was made
            msg = 'Too slow! Please respond faster.'
        else:
            msg = ''
        
        
        text.setColor([1.0000, -1.0000, -1.0000], colorSpace='rgb')
        text.setText(msg)
        # store start times for error_feedback
        error_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        error_feedback.tStart = globalClock.getTime(format='float')
        error_feedback.status = STARTED
        thisExp.addData('error_feedback.started', error_feedback.tStart)
        error_feedback.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        error_feedbackComponents = error_feedback.components
        for thisComponent in error_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "error_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trial_loop3, data.TrialHandler2) and thisTrial_loop3.thisN != trial_loop3.thisTrial.thisN:
            continueRoutine = False
        error_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                error_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in error_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "error_feedback" ---
        for thisComponent in error_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for error_feedback
        error_feedback.tStop = globalClock.getTime(format='float')
        error_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('error_feedback.stopped', error_feedback.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if error_feedback.maxDurationReached:
            routineTimer.addTime(-error_feedback.maxDuration)
        elif error_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'trial_loop3'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[text_norm_2, key_instruct_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_2
    key_instruct_2.keys = []
    key_instruct_2.rt = []
    _key_instruct_2_allKeys = []
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm_2* updates
        
        # if text_norm_2 is starting this frame...
        if text_norm_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm_2.frameNStart = frameN  # exact frame index
            text_norm_2.tStart = t  # local t and not account for scr refresh
            text_norm_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm_2.status = STARTED
            text_norm_2.setAutoDraw(True)
        
        # if text_norm_2 is active this frame...
        if text_norm_2.status == STARTED:
            # update params
            pass
        
        # *key_instruct_2* updates
        waitOnFlip = False
        
        # if key_instruct_2 is starting this frame...
        if key_instruct_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_2.frameNStart = frameN  # exact frame index
            key_instruct_2.tStart = t  # local t and not account for scr refresh
            key_instruct_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct_2.started')
            # update status
            key_instruct_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_2.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_2.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_2_allKeys.extend(theseKeys)
            if len(_key_instruct_2_allKeys):
                key_instruct_2.keys = _key_instruct_2_allKeys[0].name  # just the first key pressed
                key_instruct_2.rt = _key_instruct_2_allKeys[0].rt
                key_instruct_2.duration = _key_instruct_2_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # check responses
    if key_instruct_2.keys in ['', [], None]:  # No response was made
        key_instruct_2.keys = None
    thisExp.addData('key_instruct_2.keys',key_instruct_2.keys)
    if key_instruct_2.keys != None:  # we had a response
        thisExp.addData('key_instruct_2.rt', key_instruct_2.rt)
        thisExp.addData('key_instruct_2.duration', key_instruct_2.duration)
    thisExp.nextEntry()
    # the Routine "break_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_loop4 = data.TrialHandler2(
        name='trial_loop4',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trial_loop4)  # add the loop to the experiment
    thisTrial_loop4 = trial_loop4.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop4.rgb)
    if thisTrial_loop4 != None:
        for paramName in thisTrial_loop4:
            globals()[paramName] = thisTrial_loop4[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_loop4 in trial_loop4:
        currentLoop = trial_loop4
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop4.rgb)
        if thisTrial_loop4 != None:
            for paramName in thisTrial_loop4:
                globals()[paramName] = thisTrial_loop4[paramName]
        
        # --- Prepare to start Routine "trials" ---
        # create an object to store info about Routine trials
        trials = data.Routine(
            name='trials',
            components=[fixation, leftBox, rightBox, key_resp, cue, target],
        )
        trials.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        cue.setPos(cueSide)
        target.setPos(targetSide)
        # store start times for trials
        trials.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trials.tStart = globalClock.getTime(format='float')
        trials.status = STARTED
        thisExp.addData('trials.started', trials.tStart)
        trials.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        trialsComponents = trials.components
        for thisComponent in trials.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trials" ---
        # if trial has changed, end Routine now
        if isinstance(trial_loop4, data.TrialHandler2) and thisTrial_loop4.thisN != trial_loop4.thisTrial.thisN:
            continueRoutine = False
        trials.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 0.8+ISI+ 1-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *leftBox* updates
            
            # if leftBox is starting this frame...
            if leftBox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                leftBox.frameNStart = frameN  # exact frame index
                leftBox.tStart = t  # local t and not account for scr refresh
                leftBox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(leftBox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'leftBox.started')
                # update status
                leftBox.status = STARTED
                leftBox.setAutoDraw(True)
            
            # if leftBox is active this frame...
            if leftBox.status == STARTED:
                # update params
                leftBox.setLineColor([-1.0000, -1.0000, -1.0000], log=False)
            
            # if leftBox is stopping this frame...
            if leftBox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > leftBox.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    leftBox.tStop = t  # not accounting for scr refresh
                    leftBox.tStopRefresh = tThisFlipGlobal  # on global time
                    leftBox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'leftBox.stopped')
                    # update status
                    leftBox.status = FINISHED
                    leftBox.setAutoDraw(False)
            
            # *rightBox* updates
            
            # if rightBox is starting this frame...
            if rightBox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rightBox.frameNStart = frameN  # exact frame index
                rightBox.tStart = t  # local t and not account for scr refresh
                rightBox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rightBox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rightBox.started')
                # update status
                rightBox.status = STARTED
                rightBox.setAutoDraw(True)
            
            # if rightBox is active this frame...
            if rightBox.status == STARTED:
                # update params
                rightBox.setLineColor([-1.0000, -1.0000, -1.0000], log=False)
            
            # if rightBox is stopping this frame...
            if rightBox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rightBox.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    rightBox.tStop = t  # not accounting for scr refresh
                    rightBox.tStopRefresh = tThisFlipGlobal  # on global time
                    rightBox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rightBox.stopped')
                    # update status
                    rightBox.status = FINISHED
                    rightBox.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 0.8+ISI+1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # was this correct?
                    if (key_resp.keys == str(correct)) or (key_resp.keys == correct):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *cue* updates
            
            # if cue is starting this frame...
            if cue.status == NOT_STARTED and tThisFlip >= .75-frameTolerance:
                # keep track of start time/frame for later
                cue.frameNStart = frameN  # exact frame index
                cue.tStart = t  # local t and not account for scr refresh
                cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue.started')
                # update status
                cue.status = STARTED
                cue.setAutoDraw(True)
            
            # if cue is active this frame...
            if cue.status == STARTED:
                # update params
                pass
            
            # if cue is stopping this frame...
            if cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue.tStartRefresh + .05-frameTolerance:
                    # keep track of stop time/frame for later
                    cue.tStop = t  # not accounting for scr refresh
                    cue.tStopRefresh = tThisFlipGlobal  # on global time
                    cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue.stopped')
                    # update status
                    cue.status = FINISHED
                    cue.setAutoDraw(False)
            
            # *target* updates
            
            # if target is starting this frame...
            if target.status == NOT_STARTED and tThisFlip >= 0.8+ISI-frameTolerance:
                # keep track of start time/frame for later
                target.frameNStart = frameN  # exact frame index
                target.tStart = t  # local t and not account for scr refresh
                target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target.started')
                # update status
                target.status = STARTED
                target.setAutoDraw(True)
            
            # if target is active this frame...
            if target.status == STARTED:
                # update params
                pass
            
            # if target is stopping this frame...
            if target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    target.tStop = t  # not accounting for scr refresh
                    target.tStopRefresh = tThisFlipGlobal  # on global time
                    target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target.stopped')
                    # update status
                    target.status = FINISHED
                    target.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trials.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trials.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trials" ---
        for thisComponent in trials.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trials
        trials.tStop = globalClock.getTime(format='float')
        trials.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trials.stopped', trials.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(correct).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trial_loop4 (TrialHandler)
        trial_loop4.addData('key_resp.keys',key_resp.keys)
        trial_loop4.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            trial_loop4.addData('key_resp.rt', key_resp.rt)
            trial_loop4.addData('key_resp.duration', key_resp.duration)
        # the Routine "trials" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "error_feedback" ---
        # create an object to store info about Routine error_feedback
        error_feedback = data.Routine(
            name='error_feedback',
            components=[text],
        )
        error_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        if not key_resp.keys:  # No response was made
            msg = 'Too slow! Please respond faster.'
        else:
            msg = ''
        
        
        text.setColor([1.0000, -1.0000, -1.0000], colorSpace='rgb')
        text.setText(msg)
        # store start times for error_feedback
        error_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        error_feedback.tStart = globalClock.getTime(format='float')
        error_feedback.status = STARTED
        thisExp.addData('error_feedback.started', error_feedback.tStart)
        error_feedback.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        error_feedbackComponents = error_feedback.components
        for thisComponent in error_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "error_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trial_loop4, data.TrialHandler2) and thisTrial_loop4.thisN != trial_loop4.thisTrial.thisN:
            continueRoutine = False
        error_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                error_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in error_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "error_feedback" ---
        for thisComponent in error_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for error_feedback
        error_feedback.tStop = globalClock.getTime(format='float')
        error_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('error_feedback.stopped', error_feedback.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if error_feedback.maxDurationReached:
            routineTimer.addTime(-error_feedback.maxDuration)
        elif error_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'trial_loop4'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
