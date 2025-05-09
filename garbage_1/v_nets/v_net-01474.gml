graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1474
  arrival_time 32086.866077355062
  lifetime 2.8163406307258954
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 26
    gpu 29
    rom 28
  ]
  node [
    id 1
    label "1"
    cpu 18
    gpu 29
    rom 44
  ]
  node [
    id 2
    label "2"
    cpu 12
    gpu 18
    rom 38
  ]
  node [
    id 3
    label "3"
    cpu 35
    gpu 32
    rom 12
  ]
  node [
    id 4
    label "4"
    cpu 39
    gpu 4
    rom 39
  ]
  node [
    id 5
    label "5"
    cpu 30
    gpu 47
    rom 2
  ]
  node [
    id 6
    label "6"
    cpu 19
    gpu 26
    rom 40
  ]
  node [
    id 7
    label "7"
    cpu 19
    gpu 1
    rom 25
  ]
  node [
    id 8
    label "8"
    cpu 41
    gpu 29
    rom 28
  ]
  node [
    id 9
    label "9"
    cpu 26
    gpu 40
    rom 48
  ]
  node [
    id 10
    label "10"
    cpu 43
    gpu 6
    rom 27
  ]
  edge [
    source 0
    target 1
    bw 2
  ]
  edge [
    source 1
    target 2
    bw 39
  ]
  edge [
    source 2
    target 3
    bw 32
  ]
  edge [
    source 3
    target 4
    bw 43
  ]
  edge [
    source 4
    target 5
    bw 17
  ]
  edge [
    source 5
    target 6
    bw 37
  ]
  edge [
    source 6
    target 7
    bw 25
  ]
  edge [
    source 7
    target 8
    bw 25
  ]
  edge [
    source 8
    target 9
    bw 33
  ]
  edge [
    source 9
    target 10
    bw 19
  ]
]
