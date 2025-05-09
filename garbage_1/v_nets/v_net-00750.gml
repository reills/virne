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
  id 750
  arrival_time 15899.437405737921
  lifetime 722.9129729085272
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 8
    gpu 33
    rom 30
  ]
  node [
    id 1
    label "1"
    cpu 46
    gpu 49
    rom 23
  ]
  node [
    id 2
    label "2"
    cpu 43
    gpu 5
    rom 32
  ]
  node [
    id 3
    label "3"
    cpu 30
    gpu 10
    rom 17
  ]
  node [
    id 4
    label "4"
    cpu 11
    gpu 35
    rom 47
  ]
  node [
    id 5
    label "5"
    cpu 29
    gpu 43
    rom 10
  ]
  node [
    id 6
    label "6"
    cpu 47
    gpu 25
    rom 31
  ]
  node [
    id 7
    label "7"
    cpu 41
    gpu 50
    rom 11
  ]
  node [
    id 8
    label "8"
    cpu 5
    gpu 46
    rom 44
  ]
  node [
    id 9
    label "9"
    cpu 36
    gpu 16
    rom 46
  ]
  node [
    id 10
    label "10"
    cpu 20
    gpu 12
    rom 24
  ]
  node [
    id 11
    label "11"
    cpu 35
    gpu 33
    rom 6
  ]
  node [
    id 12
    label "12"
    cpu 39
    gpu 5
    rom 41
  ]
  edge [
    source 0
    target 1
    bw 41
  ]
  edge [
    source 1
    target 2
    bw 38
  ]
  edge [
    source 2
    target 3
    bw 32
  ]
  edge [
    source 3
    target 4
    bw 4
  ]
  edge [
    source 4
    target 5
    bw 10
  ]
  edge [
    source 5
    target 6
    bw 24
  ]
  edge [
    source 6
    target 7
    bw 35
  ]
  edge [
    source 7
    target 8
    bw 24
  ]
  edge [
    source 8
    target 9
    bw 30
  ]
  edge [
    source 9
    target 10
    bw 34
  ]
  edge [
    source 10
    target 11
    bw 42
  ]
  edge [
    source 11
    target 12
    bw 28
  ]
]
