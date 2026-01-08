# Spring Boot 3.2.0 Hello World Project

This is a sample project implementing a secured Hello World API using Spring Boot 3.2.0 and Spring Security.

## Requirements
- Java 17 or higher (Required for Spring Boot 3.x)
- Maven

## Features
- **GET /hello**: Returns "Hello World". Requires authentication.
- **Security**: 
  - Username: `test`
  - Password: `123456`
  - Supports both HTTP Basic Auth and Form Login.

## How to Run
```bash
./mvnw spring-boot:run
```

## Project Structure
- `src/main/java/com/bs/HelloController.java`: The API endpoint.
- `src/main/java/com/bs/SecurityConfig.java`: Security configuration (User & FilterChain).
