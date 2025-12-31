/**
 * JPype + RPyC Example - Java Client
 *
 * Demonstrates calling RPyC from Java via JPype's Python embedding.
 *
 * Build and run:
 *   cd examples/python-bridges/jpype
 *   ./gradlew run
 *
 * Or manually:
 *   javac -cp jpype.jar RPyCClient.java
 *   java -cp .:jpype.jar RPyCClient
 */
package com.unifyweaver.bridges.jpype;

// JPype provides these imports after startJVM()
// import org.jpype.*;

public class RPyCClient {

    public static void main(String[] args) {
        System.out.println("JPype + RPyC Java Client");
        System.out.println("========================");

        // In real usage, JPype would be initialized first:
        // JPype.startJVM();

        // Then you can call Python code that uses RPyC:
        // PyObject rpyc = PyModule.import_("rpyc");
        // PyObject conn = rpyc.getAttr("classic").invoke("connect", "localhost", 18812);
        // PyObject math = conn.getAttr("modules").getAttr("math");
        // double result = math.invoke("sqrt", 16).asDouble();
        // System.out.println("sqrt(16) = " + result);
        // conn.invoke("close");

        // JPype.shutdownJVM();

        System.out.println("\nNote: This is a template. Run rpyc_client.py for the working example.");
        System.out.println("The Python script demonstrates the pattern for JPype integration.");
    }
}
