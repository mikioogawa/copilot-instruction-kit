# GitHub Copilot Instructions

This file contains instructions and guidelines for GitHub Copilot to help generate better code suggestions in this project.

## Project Overview

[Provide a brief description of your project, its purpose, and main technologies used]

## Code Style and Conventions

### General Guidelines
- Follow consistent naming conventions throughout the codebase
- Write clear, self-documenting code with meaningful variable and function names
- Keep functions small and focused on a single responsibility
- Add comments for complex logic or non-obvious implementations

### Language-Specific Guidelines

#### JavaScript/TypeScript
- Use ES6+ syntax and modern JavaScript features
- Prefer `const` over `let`, avoid `var`
- Use arrow functions for anonymous functions
- Use template literals for string interpolation
- Follow consistent indentation (spaces vs tabs)

#### Python
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write descriptive docstrings for functions and classes
- Use list comprehensions for simple iterations

#### Other Languages
[Add guidelines specific to the languages used in your project]

## Project Structure

[Describe the directory structure and organization of your project]

```
project-root/
├── src/           # Source code
├── tests/         # Test files
├── docs/          # Documentation
└── config/        # Configuration files
```

## Testing Guidelines

- Write unit tests for all new features
- Maintain high test coverage
- Use descriptive test names that explain what is being tested
- Follow the Arrange-Act-Assert (AAA) pattern in tests

## Documentation Standards

- Document all public APIs and exported functions
- Keep README.md up to date with project changes
- Include examples in documentation where appropriate
- Document environment variables and configuration options

## Dependencies and Imports

- Group imports logically (standard library, third-party, local)
- Avoid circular dependencies
- Use specific imports rather than wildcard imports
- Keep dependencies up to date

## Error Handling

- Use appropriate error handling mechanisms (try-catch, error types)
- Provide meaningful error messages
- Log errors appropriately
- Handle edge cases gracefully

## Security Best Practices

- Never commit sensitive information (API keys, passwords, tokens)
- Validate and sanitize all user inputs
- Use parameterized queries to prevent SQL injection
- Follow principle of least privilege

## Performance Considerations

- Consider time and space complexity
- Avoid premature optimization
- Profile code before optimizing
- Cache expensive operations when appropriate

## Git Workflow

- Write clear, descriptive commit messages
- Keep commits focused and atomic
- Reference issue numbers in commits when applicable
- Follow branch naming conventions

## Additional Guidelines

[Add any project-specific guidelines, patterns, or preferences here]

## Examples

### Good Code Example
```javascript
// Clear, descriptive function with proper error handling
function calculateUserAge(birthDate) {
  if (!birthDate || !(birthDate instanceof Date)) {
    throw new Error('Invalid birth date provided');
  }
  
  const today = new Date();
  const age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
    return age - 1;
  }
  
  return age;
}
```

### Code to Avoid
```javascript
// Unclear variable names and no error handling
function calc(d) {
  let t = new Date();
  let a = t.getFullYear() - d.getFullYear();
  return a;
}
```

## Resources

- [Project Documentation](link-to-docs)
- [Contributing Guidelines](link-to-contributing)
- [Code Review Checklist](link-to-checklist)

## Notes for Copilot

When generating code suggestions:
1. Always consider the existing codebase patterns
2. Prioritize readability and maintainability over cleverness
3. Include appropriate error handling
4. Follow the established code style
5. Generate relevant tests alongside new features

---

*This file should be customized based on your project's specific needs and conventions.*
