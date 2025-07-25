..  SPDX-License-Identifier: BSD-3-Clause
    Copyright(c) 2010-2014 Intel Corporation.

QoS Scheduler Sample Application
================================

The QoS sample application demonstrates the use of the DPDK to provide QoS scheduling.

Overview
--------

The architecture of the QoS scheduler application is shown in the following figure.

.. _figure_qos_sched_app_arch:

.. figure:: img/qos_sched_app_arch.*

   QoS Scheduler Application Architecture


There are two flavors of the runtime execution for this application,
with two or three threads per each packet flow configuration being used.
The RX thread reads packets from the RX port,
classifies the packets based on the double VLAN (outer and inner) and
the lower byte of the IP destination address and puts them into the ring queue.
The worker thread dequeues the packets from the ring and calls the QoS scheduler enqueue/dequeue functions.
If a separate TX core is used, these are sent to the TX ring.
Otherwise, they are sent directly to the TX port.
The TX thread, if present, reads from the TX ring and write the packets to the TX port.

Compiling the Application
-------------------------

To compile the sample application see :doc:`compiling`.

The application is located in the ``qos_sched`` sub-directory.

    .. note::

        This application is intended as a linux only.

.. note::

    Number of grinders is currently set to 8.
    This can be modified by specifying RTE_SCHED_PORT_N_GRINDERS=N
    in CFLAGS, where N is number of grinders.

Running the Application
-----------------------

.. note::

    In order to run the application, a total of at least 4
    G of huge pages must be set up for each of the used sockets (depending on the cores in use).

The application has a number of command line options:

.. code-block:: console

    ./<build_dir>/examples/dpdk-qos_sched [EAL options] -- <APP PARAMS>

Mandatory application parameters include:

*   --pfc "RX PORT, TX PORT, RX LCORE, WT LCORE, TX CORE": Packet flow configuration.
    Multiple pfc entities can be configured in the command line,
    having 4 or 5 items (if TX core defined or not).

Optional application parameters include:

*   -i: It makes the application to start in the interactive mode.
    In this mode, the application shows a command line that can be used for obtaining statistics while
    scheduling is taking place (see interactive mode below for more information).

*   --mnc n: Main core index (the default value is 1).

*   --rsz "A, B, C": Ring sizes:

*   A = Size (in number of buffer descriptors) of each of the NIC RX rings read
    by the I/O RX lcores (the default value is 128).

*   B = Size (in number of elements) of each of the software rings used
    by the I/O RX lcores to send packets to worker lcores (the default value is 8192).

*   C = Size (in number of buffer descriptors) of each of the NIC TX rings written
    by worker lcores (the default value is 256)

*   --bsz "A, B, C, D": Burst sizes

*   A = I/O RX lcore read burst size from the NIC RX (the default value is 64)

*   B = I/O RX lcore write burst size to the output software rings,
    worker lcore read burst size from input software rings,QoS enqueue size (the default value is 64)

*   C = QoS dequeue size (the default value is 63)

*   D = Worker lcore write burst size to the NIC TX (the default value is 64)

*   --msz M: Mempool size (in number of mbufs) for each pfc (default 2097152)

*   --rth "A, B, C": The RX queue threshold parameters

*   A = RX prefetch threshold (the default value is 8)

*   B = RX host threshold (the default value is 8)

*   C = RX write-back threshold (the default value is 4)

*   --tth "A, B, C": TX queue threshold parameters

*   A = TX prefetch threshold (the default value is 36)

*   B = TX host threshold (the default value is 0)

*   C = TX write-back threshold (the default value is 0)

*   --cfg FILE: Profile configuration to load

Refer to *DPDK Getting Started Guide* for general information on running applications and
the Environment Abstraction Layer (EAL) options.

The profile configuration file defines all the port/subport/pipe/traffic class/queue parameters
needed for the QoS scheduler configuration.

The profile file has the following format:

.. literalinclude:: ../../../examples/qos_sched/profile.cfg
    :start-after: Data Plane Development Kit (DPDK) Programmer's Guide

Interactive mode
~~~~~~~~~~~~~~~~

These are the commands that are currently working under the command line interface:

*   Control Commands

*   --quit: Quits the application.

*   General Statistics

    *   stats app: Shows a table with in-app calculated statistics.

    *   stats port X subport Y: For a specific subport, it shows the number of packets that
        went through the scheduler properly and the number of packets that were dropped.
        The same information is shown in bytes.
        The information is displayed in a table separating it in different traffic classes.

    *   stats port X subport Y pipe Z: For a specific pipe, it shows the number of packets that
        went through the scheduler properly and the number of packets that were dropped.
        The same information is shown in bytes.
        This information is displayed in a table separating it in individual queues.

*   Average queue size

All of these commands work the same way, averaging the number of packets throughout a specific subset of queues.

Two parameters can be configured for this prior to calling any of these commands:

    *   qavg n X: n is the number of times that the calculation will take place.
        Bigger numbers provide higher accuracy. The default value is 10.

    *   qavg period X: period is the number of microseconds that will be allowed between each calculation.
        The default value is 100.

The commands that can be used for measuring average queue size are:

*   qavg port X subport Y: Show average queue size per subport.

*   qavg port X subport Y tc Z: Show average queue size per subport for a specific traffic class.

*   qavg port X subport Y pipe Z: Show average queue size per pipe.

*   qavg port X subport Y pipe Z tc A: Show average queue size per pipe for a specific traffic class.

*   qavg port X subport Y pipe Z tc A q B: Show average queue size of a specific queue.

Example
~~~~~~~

The following is an example command with a single packet flow configuration:

.. code-block:: console

    ./<build_dir>/examples/dpdk-qos_sched -l 1,5,7 -n 4 -- --pfc "3,2,5,7" --cfg ./profile.cfg

This example uses a single packet flow configuration which creates one RX thread on lcore 5 reading
from port 3 and a worker thread on lcore 7 writing to port 2.

Another example with 2 packet flow configurations using different ports but sharing the same core for QoS scheduler is given below:

.. code-block:: console

   ./<build_dir>/examples/dpdk-qos_sched -l 1,2,6,7 -n 4 -- --pfc "3,2,2,6,7" --pfc "1,0,2,6,7" --cfg ./profile.cfg

Note that independent cores for the packet flow configurations for each of the RX, WT and TX thread are also supported,
providing flexibility to balance the work.

The EAL corelist is constrained to contain the default main core 1 and the RX, WT and TX cores only.

Explanation
-----------

The Port/Subport/Pipe/Traffic Class/Queue are the hierarchical entities in a typical QoS application:

*   A subport represents a predefined group of users.

*   A pipe represents an individual user/subscriber.

*   A traffic class is the representation of a different traffic type with a specific loss rate,
    delay and jitter requirements; such as data voice, video or data transfers.

*   A queue hosts packets from one or multiple connections of the same type belonging to the same user.

The traffic flows that need to be configured are application dependent.
This application classifies based on the QinQ double VLAN tags and the IP destination address as indicated in the following table.

.. _table_qos_scheduler_1:

.. table:: Entity Types

   +----------------+-------------------------+--------------------------------------------------+----------------------------------+
   | **Level Name** | **Siblings per Parent** | **QoS Functional Description**                   | **Selected By**                  |
   |                |                         |                                                  |                                  |
   +================+=========================+==================================================+==================================+
   | Port           | -                       | Ethernet port                                    | Physical port                    |
   |                |                         |                                                  |                                  |
   +----------------+-------------------------+--------------------------------------------------+----------------------------------+
   | Subport        | Config (8)              | Traffic shaped (token bucket)                    | Outer VLAN tag                   |
   |                |                         |                                                  |                                  |
   +----------------+-------------------------+--------------------------------------------------+----------------------------------+
   | Pipe           | Config (4k)             | Traffic shaped (token bucket)                    | Inner VLAN tag                   |
   |                |                         |                                                  |                                  |
   +----------------+-------------------------+--------------------------------------------------+----------------------------------+
   | Traffic Class  | 13                      | TCs of the same pipe services in strict priority | Destination IP address (0.0.0.X) |
   |                |                         |                                                  |                                  |
   +----------------+-------------------------+--------------------------------------------------+----------------------------------+
   | Queue          | High Priority TC: 1,    | Queue of lowest priority traffic                 | Destination IP address (0.0.0.X) |
   |                | Lowest Priority TC: 4   | class (Best effort) serviced in WRR              |                                  |
   +----------------+-------------------------+--------------------------------------------------+----------------------------------+

Please refer to the "QoS Scheduler" chapter in the *DPDK Programmer's Guide* for more information about these parameters.
