// Azure OpenAI — referenced as an EXISTING account, never (re)created.
//
// Why `existing`:
//   * The live account `aoai-askgloucester-dev` was first deployed with a
//     customSubDomainName carrying a uniqueString() suffix — its endpoint is
//     https://aoai-askgloucester-dev-7c8ac.openai.azure.com/. customSubDomainName
//     is IMMUTABLE, and this module can't reproduce that exact suffix, so the
//     account can't be redeployed in place without changing (breaking) the
//     endpoint the app already uses.
//   * The subscription caps S0 OpenAI accounts at 1 per region
//     (OpenAI.S0.AccountCount = 1/1 in eastus), so a second account can't be
//     created either.
// We therefore reference the account and only manage its model deployments,
// which ARE updatable, as child resources. The endpoint consumed by other
// modules comes from the existing account's own properties (never hardcoded).

@description('Name of the existing Azure OpenAI account.')
param name string

@description('Principal ID of the managed identity to grant access to.')
param principalId string

@description('Object ID of the GitHub Actions CI/CD service principal to grant access to.')
param githubActionsSpObjectId string

@description('Name of the embedding model deployment.')
param embeddingDeploymentName string = 'text-embedding-3-small'

@description('Name of the chat model deployment.')
param chatDeploymentName string = 'gpt-4.1-mini'

// Built-in role: Cognitive Services OpenAI User
var cognitiveServicesOpenAiUserRoleId = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'

// The pre-existing account. No location/sku/properties here — we don't own its
// lifecycle, only its deployments and RBAC.
resource openAi 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: name
}

// text-embedding-3-small — Standard, v1. capacity is TPM in thousands; raised
// from 120 (120K TPM) to 350 (350K TPM) so a full reindex isn't throttled by the
// embedding TPM ceiling (the source of the 59s rate-limit waits). Model, version
// and dimensions are unchanged.
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAi
  name: embeddingDeploymentName
  sku: {
    name: 'Standard'
    capacity: 350
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-3-small'
      version: '1'
    }
    // Pinned to live defaults so what-if/deploy don't show a phantom property
    // flip on these already-existing properties.
    raiPolicyName: 'Microsoft.DefaultV2'
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
  }
}

// gpt-4.1-mini — GlobalStandard, 100K TPM, version 2025-04-14. Already live;
// declared here to bring the chat deployment under IaC.
resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAi
  name: chatDeploymentName
  sku: {
    name: 'GlobalStandard'
    capacity: 100
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4.1-mini'
      version: '2025-04-14'
    }
    // Pinned to live defaults (see embedding deployment above).
    raiPolicyName: 'Microsoft.DefaultV2'
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
  }
  // Cognitive Services rejects concurrent deployment operations on one account,
  // so serialize this after the embedding deployment.
  dependsOn: [
    embeddingDeployment
  ]
}

resource openAiUserAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openAi.id, principalId, cognitiveServicesOpenAiUserRoleId)
  scope: openAi
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAiUserRoleId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

// CI/CD: the GitHub Actions service principal needs OpenAI access so the
// scheduled ingestion workflow can generate embeddings for the documents.
resource githubOpenAiUserAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openAi.id, githubActionsSpObjectId, cognitiveServicesOpenAiUserRoleId)
  scope: openAi
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAiUserRoleId)
    principalId: githubActionsSpObjectId
    principalType: 'ServicePrincipal'
  }
}

@description('Name of the Azure OpenAI account.')
output name string = openAi.name

@description('Azure OpenAI endpoint (from the existing account properties).')
output endpoint string = openAi.properties.endpoint

@description('Name of the embedding model deployment.')
output embeddingDeploymentName string = embeddingDeployment.name

@description('Name of the chat model deployment.')
output chatDeploymentName string = chatDeployment.name
