using UnrealBuildTool;

public class IOR3ClientTarget : TargetRules
{
	public IOR3ClientTarget(TargetInfo Target) : base(Target)
	{
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		Type = TargetType.Client;
		ExtraModuleNames.Add("IOR3");
	}
}
