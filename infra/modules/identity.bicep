// User-assigned managed identity used by AskGloucester services for RBAC access.

@description('Name of the user-assigned managed identity.')
param name string

@description('Azure region for the identity.')
param location string

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: name
  location: location
}

@description('Resource ID of the managed identity.')
output id string = identity.id

@description('Name of the managed identity.')
output name string = identity.name

@description('AAD principal (object) ID used for role assignments.')
output principalId string = identity.properties.principalId

@description('Client ID of the managed identity.')
output clientId string = identity.properties.clientId
