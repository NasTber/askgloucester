// Hosting for the AskGloucester FastAPI service.
//
// Provisions, in one module to keep the app's runtime plane together:
//   1. a Log Analytics workspace (the Container Apps env logs into it),
//   2. a Consumption-plan Container Apps managed environment,
//   3. an Azure Container Registry (Basic, admin disabled — pull is via RBAC),
//   4. the Container App itself, running under the existing user-assigned
//      managed identity, and
//   5. an AcrPull role assignment so that identity can pull the image.
//
// Like the data/AI modules, all access is keyless: the Container App uses the
// user-assigned identity for both the registry pull and (at runtime, via
// DefaultAzureCredential) the Azure data services.

@description('Azure region for all resources in this module.')
param location string

@description('Name of the Container App.')
param containerAppName string

@description('Name of the Container Apps managed environment.')
param managedEnvironmentName string

@description('Name of the Log Analytics workspace backing the environment.')
param logAnalyticsName string

@description('Name of the Azure Container Registry (5-50 alphanumeric chars).')
param containerRegistryName string

@description('Resource ID of the existing user-assigned managed identity.')
param identityResourceId string

@description('Principal (object) ID of the managed identity, for role assignments.')
param identityPrincipalId string

@description('Client ID of the managed identity, surfaced to DefaultAzureCredential.')
param identityClientId string

@description('Object ID of the GitHub Actions CI/CD service principal to grant access to.')
param githubActionsSpObjectId string

@description('Container image reference for the API.')
param containerImage string

// --- App configuration (plain values, not secrets) -------------------------
@description('Storage account name (AZURE_STORAGE_ACCOUNT_NAME).')
param storageAccountName string

@description('Raw documents container name (RAW_DOCUMENTS_CONTAINER).')
param rawDocumentsContainer string

@description('Document Intelligence endpoint (AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT).')
param documentIntelligenceEndpoint string

@description('Azure AI Search endpoint (AZURE_SEARCH_ENDPOINT).')
param searchEndpoint string

@description('Azure AI Search index name (AZURE_SEARCH_INDEX_NAME).')
param searchIndexName string

@description('Azure OpenAI endpoint (AZURE_OPENAI_ENDPOINT).')
param openAiEndpoint string

@description('Azure OpenAI embedding deployment name (AZURE_OPENAI_EMBEDDING_DEPLOYMENT).')
param openAiEmbeddingDeployment string

@description('Azure OpenAI chat deployment name (AZURE_OPENAI_CHAT_DEPLOYMENT).')
param openAiChatDeployment string

@description('Azure OpenAI API version (AZURE_OPENAI_API_VERSION).')
param openAiApiVersion string

// Built-in role: AcrPull
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
// Built-in role: AcrPush — the CI/CD service principal pushes the API image.
var acrPushRoleId = '8311e382-0749-4cb8-b61a-304f252e45ec'
// Built-in role: Contributor — the CI/CD service principal runs
// `az containerapp update` to roll the app to a new image.
var contributorRoleId = 'b24988ac-6180-42a0-ab88-20f7382dd24c'

// 1. Log Analytics workspace — the Container Apps environment ships its
//    application/system logs here.
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// 2. Container Apps managed environment (Consumption plan — no workloadProfiles
//    block means the default Consumption-only environment). Logs flow to the
//    workspace above via its customer ID + shared key.
resource managedEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: managedEnvironmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        // listKeys() reads the workspace's shared key at deploy time so the
        // environment can authenticate to Log Analytics. Not exposed as output.
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// 3. Azure Container Registry. Admin user is disabled — the Container App pulls
//    with the managed identity (AcrPull below), never registry credentials.
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

// 4. AcrPull on the registry -> the managed identity, so the Container App can
//    pull the image once the real one is built and pushed.
resource acrPullAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerRegistry.id, identityPrincipalId, acrPullRoleId)
  scope: containerRegistry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// CI/CD: AcrPush on the registry -> the GitHub Actions service principal, so the
// deploy workflow can push the built API image to this ACR.
resource acrPushAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerRegistry.id, githubActionsSpObjectId, acrPushRoleId)
  scope: containerRegistry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPushRoleId)
    principalId: githubActionsSpObjectId
    principalType: 'ServicePrincipal'
  }
}

// 5. The Container App.
//
// IMPORTANT: `containerImage` is a PLACEHOLDER at first deploy (e.g.
// mcr.microsoft.com/azuredocs/containerapps-helloworld:latest) because the real
// API image hasn't been built/pushed to ACR yet. After the first `az acr build`
// into this registry, redeploy (or `az containerapp update`) with the real
// image reference — `${containerRegistry.properties.loginServer}/askgloucester-api:<tag>`.
// The registries block below already wires identity-based pull from this ACR,
// so the only thing that changes later is the image string.
resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identityResourceId}': {}
    }
  }
  properties: {
    managedEnvironmentId: managedEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      // Identity-based pull from our ACR (no admin creds). Harmless while the
      // image still points at the public MCR placeholder; required once the
      // image moves to this registry.
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: identityResourceId
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImage
          resources: {
            // 0.5 vCPU / 1Gi is a valid Consumption CPU:memory pairing. CPU is
            // fractional, so it must be passed as json('0.5'), not a bare number.
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'AZURE_STORAGE_ACCOUNT_NAME', value: storageAccountName }
            { name: 'RAW_DOCUMENTS_CONTAINER', value: rawDocumentsContainer }
            { name: 'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT', value: documentIntelligenceEndpoint }
            { name: 'AZURE_SEARCH_ENDPOINT', value: searchEndpoint }
            { name: 'AZURE_SEARCH_INDEX_NAME', value: searchIndexName }
            { name: 'AZURE_OPENAI_ENDPOINT', value: openAiEndpoint }
            { name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT', value: openAiEmbeddingDeployment }
            { name: 'AZURE_OPENAI_CHAT_DEPLOYMENT', value: openAiChatDeployment }
            { name: 'AZURE_OPENAI_API_VERSION', value: openAiApiVersion }
            // Not in the requested list, but REQUIRED: with a user-assigned
            // identity, DefaultAzureCredential must be told which identity to
            // use, otherwise the managed-identity probe is ambiguous. This is an
            // identity binding, not app config — see report note.
            { name: 'AZURE_CLIENT_ID', value: identityClientId }
          ]
          // Health probes. No readiness probe: /health is a liveness ping and
          // does NOT check Azure connectivity, so it wouldn't reflect true
          // readiness to serve traffic.
          probes: [
            {
              // Startup: tolerate the slow cold IMDS token acquisition we
              // observed. 10s delay + 30 failures x 10s = up to 300s before the
              // container is declared failed to start.
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              failureThreshold: 30
            }
            {
              // Liveness: once started, restart the container after 3 missed
              // 30s checks.
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 10
              periodSeconds: 30
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
  // The app must not be created before the identity can pull from ACR.
  dependsOn: [
    acrPullAssignment
  ]
}

// CI/CD: Contributor on the Container App -> the GitHub Actions service
// principal, so the deploy workflow can run `az containerapp update` to roll the
// app to a newly pushed image.
resource containerAppContributorAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerApp.id, githubActionsSpObjectId, contributorRoleId)
  scope: containerApp
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', contributorRoleId)
    principalId: githubActionsSpObjectId
    principalType: 'ServicePrincipal'
  }
}

@description('Name of the Container App.')
output name string = containerApp.name

@description('Public FQDN of the Container App ingress.')
output fqdn string = containerApp.properties.configuration.ingress.fqdn

@description('Name of the Azure Container Registry.')
output containerRegistryName string = containerRegistry.name

@description('Login server of the Azure Container Registry (push/pull host).')
output containerRegistryLoginServer string = containerRegistry.properties.loginServer
