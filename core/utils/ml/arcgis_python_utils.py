# coding: utf-8
import arcpy


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "ClipTool"
        self.alias = "ClipTool"

        # List of tool classes associated with this toolbox
        self.tools = [ClipTool]


class ClipTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "ClipTool"
        self.description = "ClipTool"
        self.canRunInBackground = False

    def getParameterInfo(self):
        # Input feature class
        param0 = arcpy.Parameter(
            displayName="in_features",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")

        # Input table
        param1 = arcpy.Parameter(
            displayName="Clip Features",
            name="clip_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")

        # Input workspace
        param2 = arcpy.Parameter(
            displayName="Output Feature Class",
            name="out_feature_class",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output")

        # Set the dependencies for the output and its schema properties
        #  The two input parameters are feature classes.
        #
        param2.parameterDependencies = [param0.name, param1.name]

        # Feature type, geometry type, and fields all come from the first
        #  dependency (parameter 0), the input features
        #
        param2.schema.featureTypeRule = "FirstDependency"
        param2.schema.geometryTypeRule = "FirstDependency"
        param2.schema.fieldsRule = "FirstDependency"

        # The extent of the output is the intersection of the input features
        #  and the clip features (parameter 1)
        #
        param2.schema.extentRule = "Intersection"

        params = [param0, param1, param2]

        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        inFeatures = parameters[0].valueAsText
        clipFeatures = parameters[1].valueAsText
        outFeatureClass = parameters[2].valueAsText

        if int(arcpy.GetCount_management(inFeatures)[0]) == 0:
            messages.addErrorMessage("{0} has no features.".format(inFeatures))
            raise arcpy.ExecuteError
        if int(arcpy.GetCount_management(clipFeatures)[0]) == 0:
            messages.addErrorMessage("{0} has no features.".format(clipFeatures))
            raise arcpy.ExecuteError

        arcpy.AddMessage("inFeatures=" + inFeatures)
        arcpy.AddMessage("clipFeatures =" + clipFeatures)
        arcpy.AddMessage("outFeatureClass =" + outFeatureClass)
        arcpy.Clip_analysis(inFeatures, clipFeatures, outFeatureClass)

        return
